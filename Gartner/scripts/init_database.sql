/*
=============================================================
Create Database and Schemas
=============================================================
Script Purpose:
    This script creates a new database named 'GartnerDB' after checking if it already exists. 
    If the database exists, it is dropped and recreated. Additionally, the script sets up three schemas 
    within the database: 'bronze', 'silver', and 'gold'.
	
*/

USE master;
GO

-- Drop and recreate the 'GartnerDB' database
IF EXISTS (SELECT 1 FROM sys.databases WHERE name = 'GartnerDB')
BEGIN
    ALTER DATABASE GartnerDB SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
    DROP DATABASE GartnerDB;
END;
GO

-- Create the 'DataWarehouse' database
CREATE DATABASE GartnerDB;
GO

USE GartnerDB;
GO

-- Create Schemas
CREATE SCHEMA bronze;
GO

CREATE SCHEMA silver;
GO

CREATE SCHEMA gold;
GO
