// Registers the `arrow.parquet.variant` canonical Arrow type extension so
// VARIANT columns can cross the Arrow C interface. DuckDB core has no Arrow
// conversion for VARIANT and the C API has no way to register Arrow type
// extensions, hence this C++ shim. The conversion machinery lives in the
// parquet extension, so this file is only compiled when the `parquet`
// feature is enabled.
//
// Spec: https://arrow.apache.org/docs/format/CanonicalExtensions.html#parquet-variant

#include "duckdb.h"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_type_extension.hpp"
#include "duckdb/common/arrow/schema_metadata.hpp"
#include "duckdb/common/types/variant_value.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "reader/variant/variant_shredded_conversion.hpp"
#include "writer/variant_column_writer.hpp"
#include "yyjson.hpp"

namespace duckdb {

namespace {

constexpr const char *ARROW_PARQUET_VARIANT = "arrow.parquet.variant";

LogicalType GetParquetVariantStructType() {
	child_list_t<LogicalType> children;
	children.emplace_back("metadata", LogicalType::BLOB);
	children.emplace_back("value", LogicalType::BLOB);
	return LogicalType::STRUCT(std::move(children));
}

LogicalType GetUnshreddedVariantGroupType() {
	child_list_t<LogicalType> children;
	children.emplace_back("value", LogicalType::BLOB);
	return LogicalType::STRUCT(std::move(children));
}

unsafe_unique_array<char> AddSchemaName(const string &name) {
	auto name_ptr = make_unsafe_uniq_array<char>(name.size() + 1);
	for (idx_t i = 0; i < name.size(); i++) {
		name_ptr[i] = name[i];
	}
	name_ptr[name.size()] = '\0';
	return name_ptr;
}

void ReleaseVariantArrowSchema(ArrowSchema *schema) {
	if (!schema || !schema->release) {
		return;
	}
	schema->release = nullptr;
	auto holder = static_cast<DuckDBArrowSchemaHolder *>(schema->private_data);
	schema->private_data = nullptr;
	delete holder;
}

void InitializeVariantArrowChild(ArrowSchema &child, DuckDBArrowSchemaHolder &root_holder, const string &name) {
	child.private_data = nullptr;
	child.release = ReleaseVariantArrowSchema;
	child.flags = ARROW_FLAG_NULLABLE;
	root_holder.owned_type_names.push_back(AddSchemaName(name));
	child.name = root_holder.owned_type_names.back().get();
	child.n_children = 0;
	child.children = nullptr;
	child.metadata = nullptr;
	child.dictionary = nullptr;
}

string GetBinaryArrowFormat(const ClientProperties &options) {
	if (options.arrow_output_version >= ArrowFormatVersion::V1_4) {
		return "vz";
	}
	if (options.arrow_offset_size == ArrowOffsetSize::LARGE) {
		return "Z";
	}
	return "z";
}

bool IsBinaryArrowFormat(const string &format) {
	return format == "z" || format == "Z" || format == "vz";
}

void SetBinaryArrowFormat(DuckDBArrowSchemaHolder &root_holder, ArrowSchema &child, const string &format) {
	root_holder.owned_type_names.push_back(AddSchemaName(format));
	child.format = root_holder.owned_type_names.back().get();
}

void VariantToArrow(ClientContext &context, Vector &source, Vector &result, idx_t count) {
	auto transform = VariantColumnWriter::GetTransformFunction();
	auto return_type = GetParquetVariantStructType();

	vector<unique_ptr<Expression>> arguments;
	arguments.push_back(make_uniq<BoundReferenceExpression>(LogicalType::VARIANT(), 0));
	auto expr = make_uniq<BoundFunctionExpression>(return_type, transform, std::move(arguments), nullptr, false);

	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::VARIANT()}, count);
	input.data[0].Reference(source);
	input.SetCardinality(count);

	ExpressionExecutor executor(context, *expr);
	executor.SetChunk(&input);
	executor.ExecuteExpression(result);
}

void ArrowToVariant(ClientContext &context, Vector &source, Vector &result, idx_t count) {
	auto &entries = StructVector::GetEntries(source);
	auto &child_types = StructType::GetChildTypes(source.GetType());
	if (child_types.size() != entries.size()) {
		throw InternalException("Unexpected mismatch between arrow.parquet.variant struct fields and vectors");
	}

	optional_idx metadata_idx;
	optional_idx value_idx;
	for (idx_t child_idx = 0; child_idx < child_types.size(); child_idx++) {
		auto &child_name = child_types[child_idx].first;
		if (child_name == "metadata") {
			metadata_idx = child_idx;
		} else if (child_name == "value") {
			value_idx = child_idx;
		} else if (child_name == "typed_value") {
			throw NotImplementedException("arrow.parquet.variant import does not support shredded typed_value yet");
		}
	}
	if (!metadata_idx.IsValid() || !value_idx.IsValid()) {
		throw InvalidInputException("arrow.parquet.variant storage type must contain 'metadata' and 'value' fields");
	}

	auto &metadata = *entries[metadata_idx.GetIndex()];
	auto &value = *entries[value_idx.GetIndex()];

	Vector intermediate_group(GetUnshreddedVariantGroupType(), count);
	auto &group_entries = StructVector::GetEntries(intermediate_group);
	group_entries[0]->Reference(value);

	auto intermediate = VariantShreddedConversion::Convert(metadata, intermediate_group, 0, count, count);
	VariantValue::ToVARIANT(intermediate, result);
}

unique_ptr<ArrowType> GetVariantArrowType(ClientContext &context, const ArrowSchema &schema,
                                          const ArrowSchemaMetadata &schema_metadata) {
	(void)context;
	const auto format = string(schema.format);
	if (format != "+s") {
		throw InvalidInputException("arrow.parquet.variant storage type must be a struct, got format '%s'",
		                            format.c_str());
	}
	if (schema_metadata.GetOption(ArrowSchemaMetadata::ARROW_EXTENSION_NAME) != ARROW_PARQUET_VARIANT) {
		throw InvalidInputException("Expected arrow extension '%s', got '%s'", ARROW_PARQUET_VARIANT,
		                            schema_metadata.GetOption(ArrowSchemaMetadata::ARROW_EXTENSION_NAME).c_str());
	}
	if (schema.n_children < 2) {
		throw InvalidInputException("arrow.parquet.variant storage type must contain at least 'metadata' and 'value'");
	}

	optional_idx metadata_idx;
	optional_idx value_idx;
	for (idx_t child_idx = 0; child_idx < NumericCast<idx_t>(schema.n_children); child_idx++) {
		auto &child = *schema.children[child_idx];
		auto child_name = string(child.name);
		if (child_name == "metadata") {
			metadata_idx = child_idx;
		} else if (child_name == "value") {
			value_idx = child_idx;
		} else if (child_name == "typed_value") {
			throw NotImplementedException("arrow.parquet.variant import does not support shredded typed_value yet");
		}
	}
	if (!metadata_idx.IsValid() || !value_idx.IsValid()) {
		throw InvalidInputException("arrow.parquet.variant storage type must contain 'metadata' and 'value' fields");
	}
	if (!IsBinaryArrowFormat(string(schema.children[metadata_idx.GetIndex()]->format)) ||
	    !IsBinaryArrowFormat(string(schema.children[value_idx.GetIndex()]->format))) {
		throw InvalidInputException("arrow.parquet.variant 'metadata' and 'value' fields must be binary types");
	}

	return make_uniq<ArrowType>(LogicalType::VARIANT());
}

void PopulateVariantArrowSchema(DuckDBArrowSchemaHolder &root_holder, ArrowSchema &schema, const LogicalType &type,
                                ClientContext &context, const ArrowTypeExtension &extension) {
	(void)type;
	(void)extension;
	const ArrowSchemaMetadata schema_metadata = ArrowSchemaMetadata::ArrowCanonicalType(ARROW_PARQUET_VARIANT);
	root_holder.metadata_info.emplace_back(schema_metadata.SerializeMetadata());
	schema.metadata = root_holder.metadata_info.back().get();

	const auto &options = context.GetClientProperties();
	const auto binary_format = GetBinaryArrowFormat(options);

	schema.format = "+s";
	schema.n_children = 2;
	root_holder.nested_children.emplace_back();
	root_holder.nested_children.back().resize(2);
	root_holder.nested_children_ptr.emplace_back();
	root_holder.nested_children_ptr.back().resize(2);
	for (idx_t child_idx = 0; child_idx < 2; child_idx++) {
		root_holder.nested_children_ptr.back()[child_idx] = &root_holder.nested_children.back()[child_idx];
	}
	schema.children = &root_holder.nested_children_ptr.back()[0];

	InitializeVariantArrowChild(*schema.children[0], root_holder, "metadata");
	InitializeVariantArrowChild(*schema.children[1], root_holder, "value");
	schema.children[0]->flags = 0;
	SetBinaryArrowFormat(root_holder, *schema.children[0], binary_format);
	SetBinaryArrowFormat(root_holder, *schema.children[1], binary_format);
}

static string ParquetVariantBytesToJson(ClientContext &context, const_data_ptr_t metadata, idx_t metadata_len,
                                        const_data_ptr_t value, idx_t value_len) {
	Vector metadata_vector(LogicalType::BLOB, 1);
	Vector value_vector(LogicalType::BLOB, 1);
	FlatVector::GetData<string_t>(metadata_vector)[0] =
	    string_t(const_char_ptr_cast(metadata), metadata_len);
	FlatVector::GetData<string_t>(value_vector)[0] = string_t(const_char_ptr_cast(value), value_len);

	Vector intermediate_group(GetUnshreddedVariantGroupType(), 1);
	auto &group_entries = StructVector::GetEntries(intermediate_group);
	group_entries[0]->Reference(value_vector);

	auto variants = VariantShreddedConversion::Convert(metadata_vector, intermediate_group, 0, 1, 1);
	if (variants.empty() || variants[0].IsNull()) {
		return "null";
	}

	auto doc = yyjson_mut_doc_new(nullptr);
	auto json_val = variants[0].ToJSON(context, doc);
	size_t len = 0;
	char *json_cstr = yyjson_mut_val_write_opts(json_val, 0, nullptr, &len, nullptr);
	if (!json_cstr) {
		yyjson_mut_doc_free(doc);
		throw InternalException("Failed to serialize VARIANT value to JSON");
	}
	string result(json_cstr, len);
	free(json_cstr);
	yyjson_mut_doc_free(doc);
	return result;
}

} // namespace

extern "C" duckdb_state duckdb_rs_parquet_variant_bytes_to_json(duckdb_database database, const uint8_t *metadata,
                                                                idx_t metadata_len, const uint8_t *value,
                                                                idx_t value_len, char **out_json) {
	if (!database || !out_json) {
		return DuckDBError;
	}
	*out_json = nullptr;
	if (!metadata || !value) {
		return DuckDBError;
	}
	try {
		auto wrapper = reinterpret_cast<DatabaseWrapper *>(database);
		ClientContext context(wrapper->database->instance);
		auto json = ParquetVariantBytesToJson(context, metadata, metadata_len, value, value_len);
		*out_json = strdup(json.c_str());
		if (!*out_json) {
			return DuckDBError;
		}
		return DuckDBSuccess;
	} catch (...) {
		return DuckDBError;
	}
}

extern "C" duckdb_state duckdb_rs_register_variant_arrow(duckdb_database database) {
	if (!database) {
		return DuckDBError;
	}
	try {
		auto wrapper = reinterpret_cast<DatabaseWrapper *>(database);
		auto &config = DBConfig::GetConfig(*wrapper->database->instance);
		// Idempotent: skip if an extension (e.g. this one, via another
		// connection to the same database) already registered VARIANT.
		if (config.HasArrowExtension(LogicalType::VARIANT())) {
			return DuckDBSuccess;
		}
		config.RegisterArrowExtension({ARROW_PARQUET_VARIANT, PopulateVariantArrowSchema, GetVariantArrowType,
		                               make_shared_ptr<ArrowTypeExtensionData>(LogicalType::VARIANT(),
		                                                                       GetParquetVariantStructType(),
		                                                                       ArrowToVariant, VariantToArrow)});
		return DuckDBSuccess;
	} catch (...) {
		return DuckDBError;
	}
}

} // namespace duckdb
