/*
===============================================================================
Stored Procedure: Load Silver Layer (Bronze -> Silver)
===============================================================================
Script Purpose:
    This stored procedure performs the ETL (Extract, Transform, Load) process to 
    populate the 'silver' schema tables from the 'bronze' schema.
	Actions Performed:
		- Truncates Silver tables.
		- Inserts transformed and cleansed data from Bronze into Silver tables.
		
Parameters:
    None. 
	  This stored procedure does not accept any parameters or return any values.

Usage Example:
    EXEC Silver.load_silver;
===============================================================================
*/

CREATE OR ALTER PROCEDURE silver.load_silver AS
BEGIN
    DECLARE @start_time DATETIME, @end_time DATETIME, @batch_start_time DATETIME, @batch_end_time DATETIME; 
    BEGIN TRY
        SET @batch_start_time = GETDATE();
        PRINT '================================================';
        PRINT 'Loading Silver Layer';
        PRINT '================================================';

		
        SET @start_time = GETDATE();
		
		PRINT '>> Inserting Data Into: silver.market_share';
		INSERT INTO silver.market_share (
			mkt_year, 
			super_region,
			region, 
			country, 
			vendor, 
			service_1,
			service_2,
			service_3,
			vertical,
			ticker, 
			hq_cntry, 
			vendor_revenue,
			cc_revenue
		)
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
		SET @end_time = GETDATE();
        PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds';
        PRINT '>> -------------';

		
        SET @start_time = GETDATE();
		PRINT '>> Inserting Data Into: silver.market_share';
		
		INSERT INTO silver.market_share (
			mkt_year, 
			super_region,
			region, 
			country, 
			vendor, 
			service_1,
			service_2,
			service_3,
			vertical,
			ticker, 
			hq_cntry, 
			vendor_revenue,
			cc_revenue
		)
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
        SET @end_time = GETDATE();
        PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds';
        PRINT '>> -------------';

        
        SET @start_time = GETDATE();
		PRINT '>> Inserting Data Into: silver.market_share';
		INSERT INTO silver.market_share (
			mkt_year, 
			super_region,
			region, 
			country, 
			vendor,
			vendor_company,
			service_1,
			service_2,
			service_3,
			vertical,
			ticker, 
			hq_cntry, 
			vendor_revenue,
			cc_revenue
		)
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
        SET @end_time = GETDATE();
        PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds';
        PRINT '>> -------------';

        
        SET @start_time = GETDATE();
		PRINT '>> Inserting Data Into: silver.market_share';
		INSERT INTO silver.market_share (
			mkt_year, 
			super_region,
			region, 
			country, 
			vendor,
			vendor_company,
			service_1,
			service_2,
			service_3,
			vertical,
			ticker, 
			hq_cntry, 
			vendor_revenue,
			cc_revenue
		)
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
	    SET @end_time = GETDATE();
        PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds';
        PRINT '>> -------------';

		
        SET @start_time = GETDATE();
		PRINT '>> Inserting Data Into: silver.market_share';
		INSERT INTO silver.market_share (
			mkt_year, 
			super_region,
			region, 
			country, 
			vendor,
			vendor_company,
			service_1,
			service_2,
			service_3,
			vertical,
			ticker, 
			hq_cntry, 
			vendor_revenue,
			cc_revenue
		)
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
	    SET @end_time = GETDATE();
        PRINT '>> Load Duration: ' + CAST(DATEDIFF(SECOND, @start_time, @end_time) AS NVARCHAR) + ' seconds';
        PRINT '>> -------------';
		

		SET @batch_end_time = GETDATE();
		PRINT '=========================================='
		PRINT 'Loading Silver Layer is Completed';
        PRINT '   - Total Load Duration: ' + CAST(DATEDIFF(SECOND, @batch_start_time, @batch_end_time) AS NVARCHAR) + ' seconds';
		PRINT '=========================================='
		
	END TRY
	BEGIN CATCH
		PRINT '=========================================='
		PRINT 'ERROR OCCURED DURING LOADING BRONZE LAYER'
		PRINT 'Error Message' + ERROR_MESSAGE();
		PRINT 'Error Message' + CAST (ERROR_NUMBER() AS NVARCHAR);
		PRINT 'Error Message' + CAST (ERROR_STATE() AS NVARCHAR);
		PRINT '=========================================='
	END CATCH
END

EXEC silver.load_silver