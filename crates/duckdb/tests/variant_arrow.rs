//! VARIANT columns cannot cross the Arrow C interface unless an
//! `arrow.parquet.variant` type extension is registered. With the `variant-arrow`
//! feature, duckdb-rs registers that extension on open,
//! so VARIANT columns surface as the canonical struct of `metadata`/`value`
//! binaries.
#![cfg(feature = "variant-arrow")]

use std::sync::Arc;

use arrow::array::{Array, BinaryArray, StructArray};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::record_batch::RecordBatch;
use duckdb::{Connection, Result};

fn setup() -> Result<Connection> {
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(
        "CREATE TABLE t (id INT, attributes VARIANT);
         INSERT INTO t VALUES (1, struct_pack(a := 1)::VARIANT);",
    )?;
    Ok(conn)
}

fn assert_variant_struct_batch(batch: &RecordBatch) {
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 2);
    let variant = batch
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("variant column should be a struct");
    let metadata = variant
        .column_by_name("metadata")
        .expect("metadata field")
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("metadata should be binary");
    let value = variant
        .column_by_name("value")
        .expect("value field")
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("value should be binary");
    assert!(!metadata.is_null(0));
    assert!(!value.is_null(0));
    assert!(!metadata.value(0).is_empty());
    assert!(!value.value(0).is_empty());
}

fn variant_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("metadata", DataType::Binary, false),
        Field::new("value", DataType::Binary, true),
    ])
}

#[test]
fn variant_query_arrow_returns_parquet_variant_struct() -> Result<()> {
    let conn = setup()?;
    let mut stmt = conn.prepare("SELECT * FROM t")?;

    let batches: Vec<RecordBatch> = stmt.query_arrow([])?.collect();

    assert_eq!(batches.len(), 1);
    assert_eq!(
        batches[0]
            .schema()
            .field(1)
            .metadata()
            .get("ARROW:extension:name")
            .map(String::as_str),
        Some("arrow.parquet.variant")
    );
    assert_variant_struct_batch(&batches[0]);
    Ok(())
}

#[test]
fn variant_stream_arrow_returns_parquet_variant_struct() -> Result<()> {
    let conn = setup()?;
    let caller_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("attributes", DataType::Struct(variant_struct_fields()), true),
    ]));
    let mut stmt = conn.prepare("SELECT * FROM t")?;

    let batches: Vec<RecordBatch> = stmt.stream_arrow([], caller_schema)?.collect::<Result<Vec<_>>>()?;

    assert_eq!(batches.len(), 1);
    assert_variant_struct_batch(&batches[0]);
    Ok(())
}

#[test]
fn variant_query_arrow_preserves_parameters() -> Result<()> {
    let conn = setup()?;
    let mut stmt = conn.prepare("SELECT * FROM t WHERE id = ?")?;

    let batches: Vec<RecordBatch> = stmt.query_arrow([1])?.collect();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    Ok(())
}

#[test]
fn variant_roundtrips_through_varchar_cast() -> Result<()> {
    let conn = setup()?;
    let json: String = conn.query_row("SELECT attributes::VARCHAR FROM t", [], |row| row.get(0))?;
    assert_eq!(json, "{'a': 1}");
    Ok(())
}

#[test]
fn variant_stream_arrow_decode_same_connection() -> Result<()> {
    let conn = setup()?;
    let caller_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("attributes", DataType::Struct(variant_struct_fields()), true),
    ]));
    let mut stmt = conn.prepare("SELECT * FROM t")?;
    let mut stream = stmt.stream_arrow([], caller_schema)?;

    let batch = stream.next().expect("expected a batch")?;

    let variant = batch
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("variant column should be a struct");
    let metadata = variant
        .column_by_name("metadata")
        .expect("metadata field")
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("metadata should be binary");
    let value = variant
        .column_by_name("value")
        .expect("value field")
        .as_any()
        .downcast_ref::<BinaryArray>()
        .expect("value should be binary");

    let json = conn.parquet_variant_bytes_to_json(metadata.value(0), value.value(0))?;
    assert_eq!(json, r#"{"a":1}"#);

    assert!(stream.next().is_none(), "stream should be exhausted");
    Ok(())
}
