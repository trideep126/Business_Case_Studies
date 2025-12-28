/*
===============================================================================
Stored Procedure: Load Bronze Layer (Source -> Bronze)
===============================================================================
Script Purpose:
    This stored procedure loads data into the 'bronze' schema from external CSV files. 
    It performs the following actions:
    - Truncates the bronze tables before loading data.
    - Uses the `BULK INSERT` command to load data from csv Files to bronze tables.

Parameters:
    None. 
	  This stored procedure does not accept any parameters or return any values.

Usage Example:
    EXEC bronze.load_bronze;
===============================================================================
*/
CREATE OR ALTER PROCEDURE bronze.load_bronze AS
BEGIN
	DECLARE @start_time DATETIME, @end_time DATETIME, @batch_start_time DATETIME, @batch_end_time DATETIME; 
	BEGIN TRY
		SET @batch_start_time = GETDATE();
		PRINT '================================================';
		PRINT 'Loading Bronze Layer';
		PRINT '================================================';

		PRINT '------------------------------------------------';
		PRINT 'Loading Market Share Tables';
		PRINT '------------------------------------------------';

		SET @start_time = GETDATE();
		PRINT '>> Truncating Table: bronze.market_2019';
		TRUNCATE TABLE bronze.market_2019;
		PRINT '>> Inserting Data Into: bronze.market_2019';
		BULK INSERT bronze.market_2019
		FROM 'C:\Users\chott\Downloads\Data Engineering\Projects\Gartner\datasets\IT_Services_Marketshare_2020Q1.CSV'
		WITH (
			FORMAT='CSV',
			FIRSTROW = 2,
			FIELDTERMINATOR = ',',
			TABLOCK
		);
		SET @end_time = GETDATE();
		PRINT '>> Load Duration: ' + CAST(DATEDIFF(second, @start_time, @end_time) AS NVARCHAR) + ' seconds';
		PRINT '>> -------------';

        SET @start_time = GETDATE();
		PRINT '>> Truncating Table: bronze.market_2020';
		TRUNCATE TABLE bronze.market_2020;

		PRINT '>> Inserting Data Into: bronze.market_2020';
		BULK INSERT bronze.market_2020
		FROM 'C:\Users\chott\Downloads\Data Engineering\Projects\Gartner\datasets\IT_Services_Marketshare_2020 (742642).CSV'
		WITH (
			FORMAT='CSV',
			FIRSTROW = 2,
			FIELDTERMINATOR = ',',
			TABLOCK
		);
		SET @end_time = GETDATE();
		PRINT '>> Load Duration: ' + CAST(DATEDIFF(second, @start_time, @end_time) AS NVARCHAR) + ' seconds';
		PRINT '>> -------------';

        SET @start_time = GETDATE();
		PRINT '>> Truncating Table: bronze.market_2021';
		TRUNCATE TABLE bronze.market_2021;
		PRINT '>> Inserting Data Into: bronze.market_2021';
		BULK INSERT bronze.market_2021
		FROM 'C:\Users\chott\Downloads\Data Engineering\Projects\Gartner\datasets\IT_Services_Marketshare_2021 (765402).CSV'
		WITH (
			FORMAT='CSV',
			FIRSTROW = 2,
			FIELDTERMINATOR = ',',
			TABLOCK
		);
		SET @end_time = GETDATE();
		PRINT '>> Load Duration: ' + CAST(DATEDIFF(second, @start_time, @end_time) AS NVARCHAR) + ' seconds';
		PRINT '>> -------------';

		
		SET @start_time = GETDATE();
		PRINT '>> Truncating Table: bronze.market_2022';
		TRUNCATE TABLE bronze.market_2022;
		PRINT '>> Inserting Data Into: bronze.market_2022';
		BULK INSERT bronze.market_2022
		FROM 'C:\Users\chott\Downloads\Data Engineering\Projects\Gartner\datasets\IT_Services_Marketshare_2022 (787876).CSV'
		WITH (
			FORMAT='CSV',
			FIRSTROW = 2,
			FIELDTERMINATOR = ',',
			TABLOCK
		);
		SET @end_time = GETDATE();
		PRINT '>> Load Duration: ' + CAST(DATEDIFF(second, @start_time, @end_time) AS NVARCHAR) + ' seconds';
		PRINT '>> -------------';

		SET @start_time = GETDATE();
		PRINT '>> Truncating Table: bronze.market_2023';
		TRUNCATE TABLE bronze.market_2023;
		PRINT '>> Inserting Data Into: bronze.market_2023';
		BULK INSERT bronze.market_2023
		FROM 'C:\Users\chott\Downloads\Data Engineering\Projects\Gartner\datasets\Services_Market_Share_2023 (808454).CSV'
		WITH (
			FORMAT='CSV',
			FIRSTROW = 2,
			FIELDTERMINATOR = ',',
			TABLOCK
		);
		SET @end_time = GETDATE();
		PRINT '>> Load Duration: ' + CAST(DATEDIFF(second, @start_time, @end_time) AS NVARCHAR) + ' seconds';
		PRINT '>> -------------';


		SET @batch_end_time = GETDATE();
		PRINT '=========================================='
		PRINT 'Loading Bronze Layer is Completed';
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
