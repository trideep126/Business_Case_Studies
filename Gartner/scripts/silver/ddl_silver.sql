/*
===============================================================================
DDL Script: Create Silver Table
===============================================================================
Script Purpose:
    This script creates a unified schema in the 'silver' layer, dropping existing table 
    if they already exist.
	  Run this script to re-define the DDL structure of 'bronze' Tables
===============================================================================
*/

IF OBJECT_ID('silver.market_share', 'U') IS NOT NULL
    DROP TABLE silver.market_share;
GO

CREATE TABLE silver.market_share (
    mkt_year VARCHAR(10), 
    super_region VARCHAR(50) NULL,
    region VARCHAR(50), 
    country VARCHAR(50), 
    vendor VARCHAR(100), 
    vendor_company VARCHAR(100) NULL, -- Added from 2021 onwards
    service_1 VARCHAR(100),
    service_2 VARCHAR(100),
    service_3 VARCHAR(100),
    vertical VARCHAR(100),
    ticker VARCHAR(50), 
    hq_cntry VARCHAR(50), 
    vendor_revenue FLOAT,
    cc_revenue FLOAT
);

/*We need to calculate market share percentages to analyze vendor dominance. */

-- Add a new column for market share %
ALTER TABLE silver.market_share
ADD market_share_percentage FLOAT;

-- Calculate Market Share by Region and Year
UPDATE silver.market_share
SET market_share_percentage = 
    CASE 
        WHEN (SELECT SUM(ISNULL(sub.vendor_revenue, 0)) 
              FROM silver.market_share AS sub 
              WHERE sub.mkt_year = silver.market_share.mkt_year 
                AND sub.region = silver.market_share.region) = 0 THEN 0
        ELSE (silver.market_share.vendor_revenue * 100.0) / 
             (SELECT SUM(ISNULL(sub.vendor_revenue, 0)) 
              FROM silver.market_share AS sub 
              WHERE sub.mkt_year = silver.market_share.mkt_year 
                AND sub.region = silver.market_share.region)
    END;

--Identify acquired vendors
/*This will return the first year when a vendor appeared under a company name.
If a vendor had NULL company name earlier but later has a company name, it means it was acquired*/

SELECT vendor, 
       vendor_company, 
       MIN(mkt_year) AS first_appearance 
FROM silver.market_share
WHERE vendor_company IS NOT NULL 
GROUP BY vendor, vendor_company;

--Create an M&A mapping table
/*Now, we will create a table to store vendor acquisitions.*/

CREATE TABLE silver.vendor_mergers (
    vendor VARCHAR(255) PRIMARY KEY, -- The acquired vendor
    acquiring_company VARCHAR(255),  -- The company that acquired it
    acquisition_year INT             -- Year of acquisition
);


--Insert acquisition data
/*This table will now store all vendors that were acquired and the year they were acquired.*/
INSERT INTO silver.vendor_mergers (vendor, acquiring_company, acquisition_year)
SELECT vendor, 
       vendor_company AS acquiring_company, 
       MIN(mkt_year) AS acquisition_year
FROM silver.market_share
WHERE vendor_company IS NOT NULL 
GROUP BY vendor, vendor_company;

select * from silver.vendor_mergers;

--Update vendor revenue to acquiring companies
/*This ensures that after an acquisition, all revenue is assigned to the acquiring company.
Before an acquisition, the vendor remains independent in the dataset.*/
UPDATE silver.market_share
SET silver.market_share.vendor_company = silver.vendor_mergers.acquiring_company
FROM silver.market_share
JOIN silver.vendor_mergers ON silver.market_share.vendor = silver.vendor_mergers.vendor
WHERE silver.market_share.mkt_year >= silver.vendor_mergers.acquisition_year;


--Verify the data
/* Acquired vendors no longer appear separately in the dataset.
Revenue has been correctly transferred to acquiring companies. */

SELECT vendor_company, 
       SUM(vendor_revenue) AS total_revenue
FROM silver.market_share
WHERE vendor_company IS NOT NULL
GROUP BY vendor_company
ORDER BY total_revenue DESC;