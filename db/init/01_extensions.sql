-- Runs once, on first start of an empty database volume.
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE SCHEMA IF NOT EXISTS raw;       -- data as ingested from sources
CREATE SCHEMA IF NOT EXISTS features;  -- cleaned / engineered features
CREATE SCHEMA IF NOT EXISTS scores;    -- model outputs and equity index
