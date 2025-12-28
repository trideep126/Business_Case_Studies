USE GartnerDB;

select * from bronze.market_2022;

--Convert '2018 YR' into integer 2018 
select 
CAST(SUBSTRING(mkt_year,1,4) as INT) as only_year
from bronze.market_2019;

--Convert  vendor_revenue into 'float' from 'varchar'
select
cast(vendor_revenue as float) as vendor_rev
from bronze.market_2019;

--Convert  cc_revenue into 'float' from 'varchar'
select
cast(cc_revenue as float) as cc_rev
from bronze.market_2019;


--Data Standardization & Normalization
select distinct service_2
from bronze.market_2019;

select distinct service_3 
from bronze.market_2019;

select
case
	when service_2 = 'Infrastructure Implementation & Managed Services' then 'IIMS'
	when service_2 = 'Business Process Services' then 'BPS'
	when service_2 = 'Application Implementation & Managed Services' then 'AIMS'
	when service_2 = 'Infrastructure as a Service' then 'IAAS'
	when service_2 = 'Hardware Support' then 'HS'
	else service_2 
end as service_2,
case
	when service_3 = 'Infrastructure Managed Services' then 'IMS'
	when service_3 = 'Application Managed Services' then 'AMS'
	when service_3 = 'Infrastructure as a Service' then 'IAAS'
	when service_3 = 'Business Process Services' then 'BPS'
	when service_3 = 'Business Consulting' then 'BC'
	when service_3 = 'Technology Consulting' then 'TC'
	when service_3 = 'Application Implementation' then 'AI'
	when service_3 = 'Infrastructure Implementation' then 'II'
	when service_3 = 'Hardware Support' then 'HS'
	else service_3 
end as service_3
from bronze.market_2021;


--Get the transformed data for 2019
SELECT 
    CAST(SUBSTRING(mkt_year,1,4) as INT) as mkt_year, 
    super_region, 
    region, 
    country, 
    vendor, 
    service_1, 
    case
		when service_2 = 'Managed Services and Cloud Infrastructure Services' then 'MSCIS'
		when service_2 = 'Business Process Outsourcing' then 'BPO'
		else service_2 
	end as service_2, 
    case
		when service_3 = 'Infrastructure Managed Services' then 'IMS'
		when service_3 = 'Application Managed Services' then 'AMS'
		when service_3 = 'Infrastructure as a Service' then 'IAAS'
		when service_3 = 'Business Process Outsourcing' then 'BPO'
		when service_3 = 'Hardware Support' then 'HS'
		else service_3 
	end as service_3, 
    vertical, 
    ticker, 
    hq_cntry, 
    CAST(vendor_revenue AS FLOAT) as vendor_revenue,
    CAST(cc_revenue AS FLOAT) as cc_revenue
FROM bronze.market_2019;


--Get transformed data for 2020
SELECT 
    CAST(SUBSTRING(mkt_year,1,4) as INT) as mkt_year, 
    super_region, 
    region, 
    country, 
    vendor, 
    service_1, 
    case
		when service_2 = 'Infrastructure Implementation & Managed Services' then 'IIMS'
		when service_2 = 'Business Process Services' then 'BPS'
		when service_2 = 'Application Implementation & Managed Services' then 'AIMS'
		when service_2 = 'Infrastructure as a Service' then 'IAAS'
		when service_2 = 'Hardware Support' then 'HS'
		else service_2 
	end as service_2, 
    case
		when service_3 = 'Infrastructure Managed Services' then 'IMS'
		when service_3 = 'Application Managed Services' then 'AMS'
		when service_3 = 'Infrastructure as a Service' then 'IAAS'
		when service_3 = 'Business Process Services' then 'BPS'
		when service_3 = 'Business Consulting' then 'BC'
		when service_3 = 'Technology Consulting' then 'TC'
		when service_3 = 'Application Implementation' then 'AI'
		when service_3 = 'Infrastructure Implementation' then 'II'
		when service_3 = 'Hardware Support' then 'HS'
		else service_3 
	end as service_3, 
    vertical, 
    ticker, 
    hq_cntry, 
    CAST(vendor_revenue AS FLOAT) as vendor_revenue,
    CAST(cc_revenue AS FLOAT) as cc_revenue
FROM bronze.market_2020;



--Get transformed data for 2021
SELECT 
    CAST(SUBSTRING(mkt_year,1,4) as INT) as mkt_year, 
    super_region, 
    region, 
    country, 
    vendor,
	vendor_company,
    service_1, 
    case
		when service_2 = 'Infrastructure Implementation & Managed Services' then 'IIMS'
		when service_2 = 'Business Process Services' then 'BPS'
		when service_2 = 'Application Implementation & Managed Services' then 'AIMS'
		when service_2 = 'Infrastructure as a Service' then 'IAAS'
		when service_2 = 'Hardware Support' then 'HS'
		else service_2 
	end as service_2, 
    case
	when service_3 = 'Infrastructure Managed Services' then 'IMS'
	when service_3 = 'Application Managed Services' then 'AMS'
	when service_3 = 'Infrastructure as a Service' then 'IAAS'
	when service_3 = 'Business Process Services' then 'BPS'
	when service_3 = 'Business Consulting' then 'BC'
	when service_3 = 'Technology Consulting' then 'TC'
	when service_3 = 'Application Implementation' then 'AI'
	when service_3 = 'Infrastructure Implementation' then 'II'
	when service_3 = 'Hardware Support' then 'HS'
	else service_3 
end as service_3, 
    vertical, 
    ticker, 
    hq_cntry, 
    CAST(vendor_revenue AS FLOAT) as vendor_revenue,
    CAST(cc_revenue AS FLOAT) as cc_revenue
FROM bronze.market_2021;


--Get transformed data for 2022
SELECT 
    CAST(SUBSTRING(mkt_year,1,4) as INT) as mkt_year, 
    super_region, 
    region, 
    country, 
    vendor,
	vendor_company,
    service_1, 
    case
		when service_2 = 'Infrastructure Implementation & Managed Services' then 'IIMS'
		when service_2 = 'Business Process Services' then 'BPS'
		when service_2 = 'Application Implementation & Managed Services' then 'AIMS'
		when service_2 = 'Infrastructure as a Service' then 'IAAS'
		when service_2 = 'Hardware Support' then 'HS'
		else service_2 
	end as service_2, 
    case
		when service_3 = 'Infrastructure Managed Services' then 'IMS'
		when service_3 = 'Application Managed Services' then 'AMS'
		when service_3 = 'Infrastructure as a Service' then 'IAAS'
		when service_3 = 'Business Process Services' then 'BPS'
		when service_3 = 'Business Consulting' then 'BC'
		when service_3 = 'Technology Consulting' then 'TC'
		when service_3 = 'Application Implementation' then 'AI'
		when service_3 = 'Infrastructure Implementation' then 'II'
		when service_3 = 'Hardware Support' then 'HS'
		else service_3 
	end as service_3, 
    vertical, 
    ticker, 
    hq_cntry, 
    CAST(vendor_revenue AS FLOAT) as vendor_revenue,
    CAST(cc_revenue AS FLOAT) as cc_revenue
FROM bronze.market_2022;


--Get the transformed data for 2023
SELECT 
    CAST(SUBSTRING(mkt_year,1,4) as INT) as mkt_year, 
    NULL AS super_region, 
    region, 
    country, 
    vendor,
	NULL AS vendor_company,
    service_1, 
    case
		when service_2 = 'Infrastructure Implementation and Managed Services' then 'IIMS'
		when service_2 = 'Business Process Services' then 'BPS'
		when service_2 = 'Application Implementation and Managed Services' then 'AIMS'
		when service_2 = 'Infrastructure as a Service (IaaS)' then 'IAAS'
		else service_2 
	end as service_2, 
    case
	when service_3 = 'Application Managed Services (AMS)' then 'AMS'
	when service_3 = 'Infrastructure Implementation' then 'II'
	when service_3 = 'Business Process Services' then 'BPS'
	when service_3 = 'Business Consulting' then 'BC'
	when service_3 = 'Technology Consulting' then 'TC'
	when service_3 = 'Application Implementation' then 'AI'
	when service_3 = 'Infrastructure Managed Services' then 'IMS'
	when service_3 = 'Infrastructure as a Service (IaaS)' then 'IAAS'
	else service_3 
end as service_3, 
    vertical, 
    ticker, 
    hq_cntry, 
    CAST(vendor_revenue AS FLOAT) as vendor_revenue,
    CAST(cc_revenue AS FLOAT) as cc_revenue
FROM bronze.market_2023;

select distinct * from silver.market_share; --2080794

select * from silver.market_share; -- 2080794
--i.e. no duplicate values

truncate table silver.market_share;

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



select * from gold.market_share_analytics;

--Check top 10 vendors by revenue
SELECT TOP 10 *
FROM gold.market_share_analytics
ORDER BY total_revenue DESC;

--Check Market Share Distribution
SELECT mkt_year, region, SUM(market_share_percentage) AS total_market_share
FROM gold.market_share_analytics
GROUP BY mkt_year, region;
