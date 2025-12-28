/*
===============================================================================
DDL Script: Create Gold Views
===============================================================================
Script Purpose:
    The Gold Layer is optimized for business intelligence (BI) tools, 
	dashboards, and reporting.

Usage:
    - These tables can be queried directly for analytics and reporting.
===============================================================================
*/

USE GartnerDB;

CREATE TABLE gold.market_share_analytics (
    mkt_year VARCHAR(10),
    region VARCHAR(50),
    country VARCHAR(50),
    vendor VARCHAR(100),
    total_revenue FLOAT,
    market_share_percentage FLOAT,
    ranking INT
);

INSERT INTO gold.market_share_analytics
SELECT 
    mkt_year, 
    region, 
    country, 
    vendor, 
    SUM(vendor_revenue) AS total_revenue,
    AVG(market_share_percentage) AS market_share_percentage,
    RANK() OVER (PARTITION BY mkt_year, region ORDER BY SUM(vendor_revenue) DESC) AS ranking
FROM silver.market_share
GROUP BY mkt_year, region, country, vendor;


/* 
1. Aggregates vendor revenue per year-region-country
2. Calculates market share %
3. Uses RANK() to rank vendors within each region
*/
