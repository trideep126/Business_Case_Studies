/*
===============================================================================
DDL Script: Create Bronze Tables
===============================================================================
Script Purpose:
    This script creates tables in the 'bronze' schema, dropping existing tables 
    if they already exist.
	  Run this script to re-define the DDL structure of 'bronze' Tables
===============================================================================
*/

IF OBJECT_ID('bronze.market_2019', 'U') IS NOT NULL
    DROP TABLE bronze.market_2019;
GO

CREATE TABLE bronze.market_2019 (
    mkt_year  NVARCHAR(50),
    super_region  NVARCHAR(50),
    region  NVARCHAR(50),
    country  NVARCHAR(50),
    vendor  NVARCHAR(50),
    service_1  NVARCHAR(50),
    service_2  NVARCHAR(50),
	service_3  NVARCHAR(50),
	vertical  NVARCHAR(50),
	ticker NVARCHAR(50),
	hq_cntry NVARCHAR(50),
	vendor_revenue  NVARCHAR(50),
	cc_revenue  NVARCHAR(50)
);
GO


IF OBJECT_ID('bronze.market_2020', 'U') IS NOT NULL
    DROP TABLE bronze.market_2020;
GO

CREATE TABLE bronze.market_2020 (
    mkt_year  NVARCHAR(50),
    super_region  NVARCHAR(50),
    region  NVARCHAR(50),
    country  NVARCHAR(50),
    vendor  NVARCHAR(50),
    service_1  NVARCHAR(50),
    service_2  NVARCHAR(50),
	service_3  NVARCHAR(50),
	vertical  NVARCHAR(50),
	ticker NVARCHAR(50),
	hq_cntry NVARCHAR(50),
	vendor_revenue  NVARCHAR(50),
	cc_revenue  NVARCHAR(50)
);
GO


IF OBJECT_ID('bronze.market_2021', 'U') IS NOT NULL
    DROP TABLE bronze.market_2021;
GO

CREATE TABLE bronze.market_2021 (
    mkt_year  NVARCHAR(50),
    super_region  NVARCHAR(50),
    region  NVARCHAR(50),
    country  NVARCHAR(50),
    vendor  NVARCHAR(50),
	vendor_company  NVARCHAR(50),
    service_1  NVARCHAR(50),
    service_2  NVARCHAR(50),
	service_3  NVARCHAR(50),
	vertical  NVARCHAR(50),
	ticker NVARCHAR(50),
	hq_cntry NVARCHAR(50),
	vendor_revenue  NVARCHAR(50),
	cc_revenue  NVARCHAR(50)
);
GO


IF OBJECT_ID('bronze.market_2022', 'U') IS NOT NULL
    DROP TABLE bronze.market_2022;
GO

CREATE TABLE bronze.market_2022 (
    mkt_year  NVARCHAR(50),
    super_region  NVARCHAR(50),
    region  NVARCHAR(50),
    country  NVARCHAR(50),
    vendor  NVARCHAR(50),
	vendor_company NVARCHAR(50),
    service_1  NVARCHAR(50),
    service_2  NVARCHAR(50),
	service_3  NVARCHAR(50),
	vertical  NVARCHAR(50),
	ticker NVARCHAR(50),
	hq_cntry NVARCHAR(50),
	vendor_revenue  NVARCHAR(50),
	cc_revenue  NVARCHAR(50)
);
GO


IF OBJECT_ID('bronze.market_2023', 'U') IS NOT NULL
    DROP TABLE bronze.market_2023;
GO

CREATE TABLE bronze.market_2023 (
    mkt_year  NVARCHAR(50),
    region  NVARCHAR(50),
    country  NVARCHAR(50),
    vendor  NVARCHAR(50),
    service_1  NVARCHAR(50),
    service_2  NVARCHAR(50),
	service_3  NVARCHAR(50),
	vertical  NVARCHAR(50),
	ticker NVARCHAR(50),
	hq_cntry NVARCHAR(50),
	vendor_revenue  NVARCHAR(50),
	cc_revenue  NVARCHAR(50)
);
GO

