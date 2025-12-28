use master

use GartnerDB;

select * from gold.market_share_analytics;
select * from silver.vendor_mergers;
select * from silver.market_share;


--Market share for key vendors over time
select vendor,
mkt_year,
market_share_percentage
from gold.market_share_analytics
where vendor in (select vendor from silver.vendor_mergers)
order by vendor,mkt_year;

--Identify change in market share over time for each vendor
with cte as (
select vendor,
mkt_year,
market_share_percentage,
lag(market_share_percentage) over(partition by vendor order by mkt_year) as previous_yr_share
from gold.market_share_analytics
), cte_2 as
(select vendor,mkt_year,
(market_share_percentage - previous_yr_share) as market_share_change
from cte 
)
select vendor,
sum(case when mkt_year=2018 then market_share_change else 0 end) as change_2018,
sum(case when mkt_year=2019 then market_share_change else 0 end) as change_2019,
sum(case when mkt_year=2020 then market_share_change else 0 end) as change_2020,
sum(case when mkt_year=2021 then market_share_change else 0 end) as change_2021,
sum(case when mkt_year=2022 then market_share_change else 0 end) as change_2022,
sum(case when mkt_year=2023 then market_share_change else 0 end) as change_2023
from cte_2
group by vendor;


--Are there any seasonal patterns in market share changes?
select mkt_year,
vendor,
avg(market_share_percentage) as avg_market_share
from gold.market_share_analytics
group by mkt_year,vendor 
order by vendor;

--How do new M&As impact vendor positions over time?
select msa.mkt_year,
case 
	when vm.acquiring_company is not null then vm.acquiring_company 
	else msa.vendor
end as consolidated_vendor,
avg(market_share_percentage) as avg_market_share
from gold.market_share_analytics as msa
left join silver.vendor_mergers vm
on msa.vendor = vm.vendor 
and msa.mkt_year >= vm.acquisition_year
where msa.vendor in (select vendor from silver.vendor_mergers)
or vm.acquiring_company is not null
group by msa.mkt_year,
case 
	when vm.acquiring_company is not null then vm.acquiring_company 
	else msa.vendor
end
order by consolidated_vendor,mkt_year;

--How much market share did vendors gain/lose after M&A?
select
case 
	when vm.acquiring_company is not null then vm.acquiring_company
	else msa.vendor 
end as consolidated_vendor,
avg(case when msa.mkt_year < vm.acquisition_year then msa.market_share_percentage else 0 end) as share_before_ma,
avg(case when msa.mkt_year >= vm.acquisition_year then msa.market_share_percentage else 0 end) as share_after_ma
from gold.market_share_analytics msa
join silver.vendor_mergers vm 
on msa.vendor = vm.vendor
group by 
case 
	when vm.acquiring_company is not null then vm.acquiring_company
	else msa.vendor 
end;

--Which years had significant revenue spikes/dips?
SELECT
    mkt_year,
    round(SUM(total_revenue),0) AS total_market_revenue,
    round(LAG(SUM(total_revenue), 1, 0) OVER (ORDER BY mkt_year),0) AS previous_year_revenue,
    round((SUM(total_revenue) - LAG(SUM(total_revenue), 1, 0) OVER (ORDER BY mkt_year)),0) AS revenue_change
FROM
    gold.market_share_analytics
GROUP BY
    mkt_year
ORDER BY
    mkt_year;

--Which region contributes the most to market share?
select region,
sum(total_revenue) as regional_revenue,
sum(total_revenue)*100 / (select sum(total_revenue) from gold.market_share_analytics where mkt_year = (select max(mkt_year) from gold.market_share_analytics)) as percentage_of_total
from gold.market_share_analytics
where mkt_year = (select max(mkt_year) from gold.market_share_analytics)
group by region
order by sum(total_revenue) desc;

--Top 5 countries across the world generating highest revenue
select top 5
country,
round(sum(total_revenue),0) as country_revenue
from gold.market_share_analytics
where mkt_year = (select max(mkt_year) from gold.market_share_analytics)
group by country 
order by sum(total_revenue) desc;

--Which vendor acquisition led to revenue growth?
WITH cte AS (
    SELECT
        vm.acquiring_company,
        vm.acquisition_year,
        SUM(CASE WHEN msa.mkt_year < vm.acquisition_year THEN msa.total_revenue ELSE 0 END) AS revenue_before_acquisition,
        SUM(CASE WHEN msa.mkt_year >= vm.acquisition_year THEN msa.total_revenue ELSE 0 END) AS revenue_after_acquisition
    FROM
        gold.market_share_analytics msa
    JOIN
        silver.vendor_mergers vm ON msa.vendor = vm.vendor
    GROUP BY
        vm.acquiring_company,
        vm.acquisition_year
    HAVING
        SUM(CASE WHEN msa.mkt_year >= vm.acquisition_year THEN msa.total_revenue ELSE 0 END) > SUM(CASE WHEN msa.mkt_year < vm.acquisition_year THEN msa.total_revenue ELSE 0 END)
), cte_2 AS (
    SELECT
        cte.acquiring_company,
        cte.acquisition_year,
        CASE
            WHEN revenue_before_acquisition <= revenue_after_acquisition THEN 'Growed'
            ELSE 'Declined'
        END AS Revenue_Growth
    FROM cte
)
SELECT count(*) 
FROM cte_2
where Revenue_Growth='Growed';

--Which services are most profitable?
select service_1,
round(sum(vendor_revenue),0) as service_revenue
from silver.market_share
group by service_1
order by sum(vendor_revenue) desc;

select service_2,
round(sum(vendor_revenue),0) as service_revenue
from silver.market_share
group by service_2
order by sum(vendor_revenue) desc;

select service_3,
round(sum(vendor_revenue),0) as service_revenue
from silver.market_share
group by service_3
order by sum(vendor_revenue) desc;

--Top 10 acquiring companies generating highest revenue
select top 10
vm.acquiring_company,
round(sum(msa.total_revenue),0) as vendor_revenue
from gold.market_share_analytics msa
join silver.vendor_mergers vm 
on msa.vendor = vm.vendor
where msa.mkt_year = (select max(mkt_year) from gold.market_share_analytics)
group by vm.acquiring_company
order by sum(msa.total_revenue) desc;

--Total Revenue contributed from specific business units
select round(sum(cc_revenue),0) from silver.market_share;
