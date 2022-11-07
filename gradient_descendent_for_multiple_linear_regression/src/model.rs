use chrono::{Datelike, NaiveDate};
use nalgebra::RowDVector;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct EstimationModel {
    pub parameters: RowDVector<f64>,
    pub b: f64,
}

#[derive(Debug)]
pub struct Gradient {
    pub parameters: RowDVector<f64>,
    pub b: f64,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct SeoulRealEstateTransactionPrice {
    #[serde(with = "datetime_ymd_format")]
    pub 계약일: NaiveDate,
    #[serde(rename(deserialize = "물건금액(만원)"))]
    pub 물건금액: Option<f64>,
    #[serde(rename(deserialize = "건물면적(㎡)"))]
    pub 건물면적: f64,
    #[serde(rename(deserialize = "토지면적(㎡)"))]
    pub 토지면적: f64,
}

impl SeoulRealEstateTransactionPrice {
    pub fn into_features(self) -> Vec<f64> {
        vec![self.건물면적]
    }

    pub fn into_target(self) -> f64 {
        self.물건금액.unwrap()
    }
}

mod datetime_ymd_format {
    use chrono::NaiveDate;
    use serde::{self, Deserialize, Deserializer};

    const FORMAT: &'static str = "%Y%m%d";

    pub fn deserialize<'de, D>(deserializer: D) -> Result<NaiveDate, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        NaiveDate::parse_from_str(&s, FORMAT).map_err(serde::de::Error::custom)
    }
}
