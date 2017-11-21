# -*- coding: utf-8 -*-
# file:   functions.py
# version:2.0.1.8
# @author: ChenKai

import warnings
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
import config
import scipy.optimize as sco
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import json
import statsmodels.api as sm
import datetime as dt
warnings.filterwarnings("ignore")
con = create_engine(
    'mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}?charset={charset}'.format(
        **config.dbconfig),encoding='utf8',echo=False,poolclass=NullPool)

   
return_noRisk = 0.03
strategies_dict = \
{'宏观策略': '宏观对冲策略',
 '管理期货': '管理期货策略',
 '股票策略': '股票多头策略',
 '股票多头': '股票多头策略',
 '事件驱动': '股票多头策略',
 '固定收益': '固定收益策略',
 '相对价值': '相对价值策略',
 '复合策略': '全市场策略',
 '组合基金': '全市场策略',
 '其它策略': '全市场策略',
 
 '混合型': '混合型',
 '联接基金': '混合型',
 '分级杠杆': '混合型',
 'ETF-场内': '混合型',
 'QDII-ETF': '混合型',
 '封闭式': '混合型',
 '其他创新': '混合型',
 'QDII': '混合型',
 'QDII-指数': '混合型',
 '债券型': '债券型',
 '定开债券': '债券型',
 '债券创新-场内': '债券型',
 '债券指数': '债券型',
 '保本型': '债券型',
 '理财型': '债券型',
 '货币型': '货币型',
 '股票指数': '股票型',
 '股票型': '股票型'}

market_dict = \
{'管理期货': '南华商品指数',
 '固定收益': '中证全债指数',
 '宏观策略': '沪深300指数',
 '股票策略': '沪深300指数',
 '股票多头': '沪深300指数',
 '事件驱动': '沪深300指数', 
 '相对价值': '沪深300指数',
 '复合策略': '沪深300指数',
 '组合基金': '沪深300指数',
 '其它策略': '沪深300指数',
 
 '混合型': '沪深300指数',
 '联接基金': '沪深300指数',
 '分级杠杆': '沪深300指数',
 'ETF-场内': '沪深300指数',
 'QDII-ETF': '沪深300指数',
 '封闭式': '沪深300指数',
 '其他创新': '沪深300指数',
 'QDII': '沪深300指数',
 'QDII-指数': '沪深300指数',
 '股票指数': '沪深300指数',
 '股票型': '沪深300指数',
 '债券型': '中证全债指数',
 '定开债券': '中证全债指数',
 '债券创新-场内': '中证全债指数',
 '债券指数': '中证全债指数',
 '保本型': '中证全债指数',
 '理财型': '中证全债指数',
 '货币型': '中证全债指数'}

#
def change_str(i):
    if i == 0:
        return False
    else:
        return True
    
def change_strnan(the_nan):
    if the_nan == 0:
        the_nan = np.nan
    return the_nan
# 从mysql中读取数据
def get_data_from_mysql(the_sql, the_index):
    the_data = pd.read_sql(the_sql, con=con, index_col=the_index)
    the_data.sort_index(inplace=True)
    con.dispose()
    return the_data


# 从mysql中industry_trend表读取数据，28列加1列时间列
def select_table_industry_trend():
    industry_trend = get_data_from_mysql('select index_date, \
        nlmy as 农林牧渔, cj as 采掘, hg as 化工, gt as 钢铁, ysjs as 有色金属, \
        dz as 电子, jydq as 家用电器, spyl as 食品饮料, fzfz as 纺织服装, \
        qgzz as 轻工制造, yysw as 医药生物, ggsy as 公共事业, jtys as 交通运输, \
        fdc as 房地产, symy as 商业贸易, xxfw as 休闲服务, zh as 综合, \
        jzcl as 建筑材料, jzzs as 建筑装饰, dqsb as 电气设备, gfjg as 国防军工, \
        jsj as 计算机, cm as 传媒, tx as 通信, yh as 银行, fyjr as 非银金融, \
        qc as 汽车, jxsb as 机械设备 from t_wind_industry_trend', 'index_date')
    industry_trend.index = pd.to_datetime(industry_trend.index)
    return industry_trend

# 从mysql中stocks_class表读取数据，14列加1列自增列
def read_data_from_sql():
    change_str = lambda i: False if i == 0 else True
    change_strnan = lambda the_nan: np.nan if the_nan == 0 else the_nan

    style_nav = select_table_style_trend()
    future_price = select_table_futures_basis()
    sw_industry_index = select_table_industry_trend()
    df_A = select_table_stocks_class()
    df_A.set_index('代码', inplace=True)  # 设置索引
    df_A['沪深300权重'] = df_A['沪深300权重'].apply(change_strnan)
    df_A['上证50权重'] = df_A['上证50权重'].apply(change_strnan)
    df_A['中证500权重'] = df_A['中证500权重'].apply(change_strnan)
    df_A['大盘股'] = df_A['大盘股'].apply(change_str)
    df_A['小盘股'] = df_A['小盘股'].apply(change_str)
    df_A['中盘股'] = df_A['中盘股'].apply(change_str)
    df_A['创业板'] = df_A['创业板'].apply(change_str)
    df_A['中小板'] = df_A['中小板'].apply(change_str)
    df_A['沪市其他'] = df_A['沪市其他'].apply(change_str)
    df_A['深市其他'] = df_A['深市其他'].apply(change_str)
    df_A['成长型'] = df_A['成长型'].apply(change_str)
    df_A['平衡型'] = df_A['平衡型'].apply(change_str)
    df_A['价值型'] = df_A['价值型'].apply(change_str)

    return style_nav, future_price, sw_industry_index, df_A

    
#
#def select_table_stocks_class():
#    stocks_class = get_data_from_mysql('select code as 代码, name as 名称, industry as 申万一级行业, \
#        market_value as 流通市值, hs300_weight as 沪深300权重, \
#        sse50_weight as 上证50权重, csi500_weight as 中证500权重, \
#        big_cap as 大盘股, small_cap as 小盘股, middle_cap as 中盘股, \
#        gem as 创业板, small_medium as 中小板, \
#        shanghai_else as 沪市其他, shenzhen_else as 深市其他 \
#        from t_wind_stocks_class', None)
#    return stocks_class

def select_table_stocks_class():
    stocks_class = get_data_from_mysql('select code as 代码, name as 名称, industry as 申万一级行业, \
        market_value as 流通市值, hs300_weight as 沪深300权重, \
        sse50_weight as 上证50权重, csi500_weight as 中证500权重, \
        big_cap as 大盘股, small_cap as 小盘股, middle_cap as 中盘股, \
        gem as 创业板, small_medium as 中小板, \
        shanghai_else as 沪市其他, shenzhen_else as 深市其他, high_growth as 成长型, low_growth as 价值型, medium_growth as 平衡型 \
        from t_wind_stocks_class', None)
    return stocks_class
# 从mysql中style_trend表读取数据，3列加1列时间列
  

def select_table_style_trend():
    style_trend = get_data_from_mysql('select index_date, \
        big_cap as 大盘股, middle_cap as 中盘股, small_cap as 小盘股 \
        from t_wind_style_trend', 'index_date')
    style_trend.index = pd.to_datetime(style_trend.index)
    return style_trend

# 从mysql中futures_basis表读取数据，6列加1列时间列


def select_table_futures_basis():
    futures_basis = get_data_from_mysql('select index_date, \
        hs300_futures as 沪深300期货, hs300 as 沪深300, \
        sse50_futures as 上证50期货, sse50 as 上证50, \
        csi500_futures as 中证500期货, csi500 as 中证500 \
        from t_wind_futures_basis', 'index_date')
    futures_basis.index = pd.to_datetime(futures_basis.index)
    return futures_basis

# 读取南华期货指数

def select_table_growth_index():
    growth_index = get_data_from_mysql('select index_date, \
        low_pb as 价值型, medium_pb as 平衡型, high_pb as 成长型 \
        from t_wind_growth_index', 'index_date')
    growth_index.index = pd.to_datetime(growth_index.index)
    return growth_index
    
    
def select_table_futures_index():
    future_index = get_data_from_mysql('select index_date ,\
        south_china_commodity as 南华期货指数 from t_wind_index_market', 'index_date')
    future_index.index = pd.to_datetime(future_index.index)
    return future_index



# 计算产品周期


def calc_cycle(nav):
    nav.dropna(inplace=True)
    week_diff = np.diff(nav.index.week)
    result = np.nanmean(np.where(week_diff < 0, np.NaN, week_diff))
    return '日' if result < 0.7 else (
        '周' if result >= 0.7 and result < 1.7 else '月')


def calc_span(days):
    if days < 91:
        return '三月以下'
    if days >= 91 and days < 182:
        return '三月以上'
    if days >= 182 and days < 365:
        return '六月以上'
    if days >= 365:
        return '一年以上'
# 收益


def calculate_profits(navs):
    pct = pd.DataFrame()
    navs = pd.DataFrame(navs)
    for i in range(len(navs.columns)):
        nav = navs.ix[:, i].dropna()
        rets = nav.pct_change().fillna(0)
        pct = pd.concat([pct, rets], axis=1, join='outer')
    pct.index.name = '日期'
    pct.columns = navs.columns
    return pct


def fetch_index(con):  # 0是日，1是周，2是月
    data = pd.read_sql(
        "select * from t_wind_index_hs300",con,index_col='index_date')
    data.columns = ['沪深300-日', '沪深300-周', '沪深300-月']
    con.dispose()
    return data


def market_index(strategy):
    # 从mysql中读取数据
    table = get_data_from_mysql('select index_date, \
                                hs300 as 沪深300指数, \
                                csi_total_debt as 中证全债指数, \
                                south_china_commodity as 南华商品指数 from t_wind_index_market', 'index_date')
    table.index = pd.to_datetime(table.index)
    return table.ix[:, 2] if strategy == '管理期货' else table.ix[:,1] if strategy == '债券策略' else table.ix[:, 0]

#def get_nav_and_info(fund_table,fund_id,start,end):
#    navs=pd.DataFrame()
#    product_info=pd.DataFrame()
#    for i in range(len(fund_id)):
#        sql_nav = 'select fund_date, netvalue,fund_id from ' + fund_table + ' where fund_id' + ' =' + str(fund_id[i])
##        sql_info1 = 'select full_name, short_name,company,strategy,mjfs,manager from ' + ('t_fund_product' if fund_table == 't_fund_netvalue' else 't_upload_product') + ' where id' + ' =' + str(fund_id[i])    
#        sql_info1 = "SELECT full_name,short_name,company,strategy,mjfs,GROUP_CONCAT(manager_name SEPARATOR ',') \
#                FROM t_fund_product ,manager_gongmu_performance WHERE fund_id="  + str(fund_id[i]) + \
#                " AND t_fund_product.id=fund_id"
#        raw_info = pd.read_sql(sql_nav, con)
#        raw_info2 = pd.read_sql(sql_info1, con)
#        navs_and_date_raw = pd.concat((v.drop_duplicates(subset='fund_date')\
#                        .set_index('fund_date').drop(labels='fund_id', axis=1) \
#                        for k, v in raw_info.groupby(by='fund_id')), axis=1)
#        
#        navs_and_date_raw.columns = raw_info2.apply(lambda x: x.short_name if x.short_name else x.full_name, axis=1)
#        navs_and_date_raw.index = pd.to_datetime(navs_and_date_raw.index)
#        navs_and_date_raw.replace(0, pd.np.nan, inplace=True)
#        navs_and_date_raw.sort_index(inplace=True)
#        navs_and_date_raw = navs_and_date_raw.loc['2012-05':]
#        navs_and_date_raw = navs_and_date_raw.loc[start:end]
#        
#        raw_info2.index = navs_and_date_raw.columns
#        products_info = pd.DataFrame()
#        products_info['Com_name'] = raw_info2.company
#        products_info['product_name'] = navs_and_date_raw.columns
#        products_info['strategy'] = raw_info2.strategy
#        products_info['product_start_date'] = navs_and_date_raw.apply(
#            lambda x: x.dropna().index[0].strftime('%Y/%m/%d'))[0]
#        products_info['product_end_date'] = navs_and_date_raw.apply(
#           lambda x: x.dropna().index[-1].strftime('%Y/%m/%d'))[0]
#        products_info['period'] = navs_and_date_raw.apply(
#           lambda x: (x.dropna().index[-1] - x.dropna().index[0]).days)[0]
#        products_info['mjfs'] = raw_info2.mjfs
#        if len(raw_info2.iloc[0,5].split(',')) <= 3:
#           products_info['manager'] = raw_info2.iloc[0,5]
#        else:
#           for i in range (3):
#              if i == 0:
#                products_info['manager'] = raw_info2.iloc[0,5].split(',')[:3][0]
#              else:
#                products_info['manager'] += ', ' + raw_info2.iloc[0,5].split(',')[:3][i]
#           products_info['manager'] += '等'  
##        products_info['manager'] = raw_info2.manager
#        products_info = products_info.fillna('暂无')
##        p = navs_and_date_raw.apply(lambda x: calc_cycle(x))
#        navs=pd.concat([navs,navs_and_date_raw],axis=1)
#        product_info=pd.concat([ product_info,products_info])
#    p=navs.apply(lambda x: calc_cycle(x))
#    return navs, product_info,p  
def get_nav_and_info(fund_table,fund_id,start,end):
    navs=pd.DataFrame()
    product_info=pd.DataFrame()
    for i in range(len(fund_id)):
        sql_nav = 'select fund_date, netvalue,fund_id from ' + fund_table + ' where fund_id' + ' =' + str(fund_id[i])
        sql_info1 = "SELECT full_name,short_name,company,strategy,mjfs,GROUP_CONCAT(manager_name SEPARATOR ',') \
                FROM t_fund_product ,manager_gongmu_performance WHERE fund_id="  + str(fund_id[i]) + \
                " AND t_fund_product.id=fund_id"
        raw_info = pd.read_sql(sql_nav, con)
        raw_info2 = pd.read_sql(sql_info1, con)
        navs_and_date_raw = pd.concat((v.drop_duplicates(subset='fund_date')\
                        .set_index('fund_date').drop(labels='fund_id', axis=1) \
                        for k, v in raw_info.groupby(by='fund_id')), axis=1)
        
        navs_and_date_raw.columns = raw_info2.apply(lambda x: x.short_name if x.short_name else x.full_name, axis=1)
        navs_and_date_raw.index = pd.to_datetime(navs_and_date_raw.index)
        navs_and_date_raw.replace(0, pd.np.nan, inplace=True)
        navs_and_date_raw.sort_index(inplace=True)
#        one_year=dt.datetime.now()-dt.timedelta(days=365)
#        one_year=one_year.strftime('%Y/%m/%d')
#        recent_time=dt.datetime.now()-dt.timedelta(days=50)
#        recent_time=recent_time.strftime('%Y/%m/%d')
#        if navs_and_date_raw.index[0].strftime('%Y/%m/%d')>one_year:           
#            return 
#        if navs_and_date_raw.index[-1].strftime('%Y/%m/%d')<recent_time:
#            return
#        navs_and_date_raw = navs_and_date_raw.loc[one_year:]
#        navs_and_date_raw = navs_and_date_raw.loc['2012-05':]
        navs_and_date_raw = navs_and_date_raw.loc[start:end]
        
        raw_info2.index = navs_and_date_raw.columns
        products_info = pd.DataFrame()
        products_info['Com_name'] = raw_info2.company
        products_info['product_name'] = navs_and_date_raw.columns
        products_info['strategy'] = raw_info2.strategy
        products_info['product_start_date'] = navs_and_date_raw.apply(
            lambda x: x.dropna().index[0].strftime('%Y/%m/%d'))[0]
        products_info['product_end_date'] = navs_and_date_raw.apply(
           lambda x: x.dropna().index[-1].strftime('%Y/%m/%d'))[0]
        products_info['period'] = navs_and_date_raw.apply(
           lambda x: (x.dropna().index[-1] - x.dropna().index[0]).days)[0]
        products_info['mjfs'] = raw_info2.mjfs
        
        if raw_info2.mjfs[0]=='公募':
           try:
                   if len(raw_info2.iloc[0,5].split(',')) <= 3:
                       products_info['manager'] = raw_info2.iloc[0,5]
                   else:
                      for i in range (3):
                         if i == 0:
                           products_info['manager'] = raw_info2.iloc[0,5].split(',')[:3][0]
                         else:
                           products_info['manager'] += ', ' + raw_info2.iloc[0,5].split(',')[:3][i]
                      products_info['manager'] += '等' 
           except:
              products_info['manager']='暂无'
        else:
             products_info['manager']='暂无'
#        products_info['manager'] = raw_info2.manager
        products_info = products_info.fillna('暂无')
#        p = navs_and_date_raw.apply(lambda x: calc_cycle(x))
        navs=pd.concat([navs,navs_and_date_raw],axis=1)
        product_info=pd.concat([ product_info,products_info])
    p=navs.apply(lambda x: calc_cycle(x))
    return navs, product_info,p          
        



def get_nav_and_info_doubletables(fund_table,fund_table2,fund_id,start,end):
    sql_nav = 'select fund_date, netvalue,fund_id from ' + fund_table + ' where fund_id=' + str(fund_id[0])
    if fund_table2 == 't_fund_product':
        sql_info1 = "SELECT full_name,short_name,company,strategy,mjfs,GROUP_CONCAT(manager_name SEPARATOR ',') \
                    FROM t_fund_product,manager_gongmu_performance WHERE fund_id="  + str(fund_id[0]) + \
                    " AND t_fund_product.id=fund_id AND is_incumbent=1"
    else:
        sql_info1 = "SELECT full_name,short_name,company,strategy,mjfs,manager FROM " + \
                    fund_table2 +" WHERE id="  + str(fund_id[0])

    raw_info = pd.read_sql(sql_nav, con)
    if len(raw_info) < 6 :
        print("error2\t\t{}".format(fund_id[0]))

    raw_info2 = pd.read_sql(sql_info1, con)
    navs_and_date_raw = pd.concat((v.drop_duplicates(subset='fund_date')\
                        .set_index('fund_date').drop(labels='fund_id', axis=1) \
                        for k, v in raw_info.groupby(by='fund_id')), axis=1)
    navs_and_date_raw.columns = raw_info2.apply(lambda x: x.short_name if x.short_name else x.full_name, axis=1)
    
    navs_and_date_raw.index = pd.to_datetime(navs_and_date_raw.index)
    navs_and_date_raw.replace(0, pd.np.nan, inplace=True)
    navs_and_date_raw.sort_index(inplace=True)
    
    #navs_and_date_raw = navs_and_date_raw.loc['2012-05':]
    navs_and_date_raw = navs_and_date_raw.loc[start:end]
    
    raw_info2.index = navs_and_date_raw.columns
    products_info = pd.DataFrame()
    products_info['Com_name'] = raw_info2.company
    products_info['product_name'] = navs_and_date_raw.columns
    products_info['strategy'] = raw_info2.strategy
    products_info['product_start_date'] = navs_and_date_raw.apply(
        lambda x: x.dropna().index[0].strftime('%Y/%m/%d'))
    products_info['product_end_date'] = navs_and_date_raw.apply(
        lambda x: x.dropna().index[-1].strftime('%Y/%m/%d'))
    products_info['period'] = navs_and_date_raw.apply(
        lambda x: (x.dropna().index[-1] - x.dropna().index[0]).days)
    products_info['mjfs'] = raw_info2.mjfs
    if products_info['mjfs'][0] == '公募':
        try:
            if len(raw_info2.iloc[0,5].split(',')) <= 3:
                products_info['manager'] = raw_info2.iloc[0,5]
            else:
                for i in range (3):
                    if i == 0:
                        products_info['manager'] = raw_info2.iloc[0,5].split(',')[:3][0]
                    else:
                        products_info['manager'] += ', ' + raw_info2.iloc[0,5].split(',')[:3][i]
                products_info['manager'] += '等'
        except:
            products_info['manager'] = '暂无'
    else:
        products_info['manager'] = '暂无'
    products_info = products_info.fillna('暂无')
    p = navs_and_date_raw.apply(lambda x: calc_cycle(x))
    return navs_and_date_raw, products_info, p


def interpolate(data, benchmark, p, strategy=None):
    if data.columns.size == 1:
        if '日' in p[0]:
            data1 = pd.DataFrame(data.dropna())
            benchmark1 = pd.DataFrame(benchmark.ix[:, 0].dropna())
        elif '周' in p[0]:
            #            data1 = pd.DataFrame(data.resample('W-FRI').ffill().dropna())
            data1 = pd.DataFrame(data.dropna())
            benchmark1 = pd.DataFrame(benchmark.ix[:, 1].dropna())
        else:
            #            data1 = pd.DataFrame(data.resample('BM').ffill().dropna())
            data1 = pd.DataFrame(data.dropna())
            benchmark1 = pd.DataFrame(benchmark.ix[:, 2].dropna())
        name = data1.columns[0]
        start_time = data1.index[0].strftime('%Y-%m-%d')
        end_time = data1.index[-1].strftime('%Y-%m-%d')
        part_benchmark = pd.DataFrame(benchmark1.ix[start_time: end_time])
        #        Merge = pd.merge(left=part_benchmark, right=data1, how='left', left_index=True, right_index=True)
        Merge = pd.concat([part_benchmark, data1], axis=1, join='outer')
        Merge[name].iloc[-1] = data1.values[-1][0]
#        df = pd.DataFrame()
#        for i, j in zip(Merge[name].dropna().index[:-1], Merge[name].dropna().index[1:]):
#            df = pd.concat([df, Merge[name][i:j].fillna(
#                (Merge[name][j] / Merge[name][i]) ** (1 / (len(Merge[name][i:j]) - 1)))[:-1].cumprod()])
#        df = pd.concat([df, Merge[name][-1:]])
        df = pd.DataFrame(Merge[name].interpolate())
        df.columns = [name]
        df2 = pd.merge(
            left=part_benchmark,
            right=df,
            how='left',
            left_index=True,
            right_index=True)
        return pd.DataFrame(df2.ix[:, 1]), pd.DataFrame(df2.ix[:, 0])
    elif strategy is None:
        period = {'日': 0, '周': 1, '月': 2, '残缺': 2}
        df2 = pd.DataFrame()
        for i, name in enumerate(data.columns):
            df = pd.DataFrame()
            if '日' in p[i]:
                data1 = pd.DataFrame(data[name].dropna())
            elif '周' in p[i]:
                data1 = pd.DataFrame(data[name].dropna())
#                data1 = pd.DataFrame(data.dropna())
            else:
                data1 = pd.DataFrame(data[name].dropna())
#                data1 = pd.DataFrame(data.dropna())
            start_time = pd.concat([benchmark.ix[:, period[p[i]]].dropna(
            ), data1], axis=1, join='inner').index[0].strftime('%Y-%m-%d')
            end_time = data1.index[-1].strftime('%Y-%m-%d')
            part_benchmark = pd.DataFrame(
                benchmark.ix[start_time: end_time, period[p[i]]].dropna())
            Merge = pd.merge(
                left=part_benchmark,
                right=data1,
                how='left',
                left_index=True,
                right_index=True)
            Merge[name].iloc[-1] = data1[name].values[-1]
            for i, j in zip(Merge[name].dropna().index[:-1],
                            Merge[name].dropna().index[1:]):
                df = pd.concat([df, Merge[name][i:j].fillna(
                    (Merge[name][j] / Merge[name][i]) ** (1 / (len(Merge[name][i:j]) - 1)))[:-1].cumprod()])
            df = pd.concat([df, Merge[name][-1:]])
            df.columns = data1.columns
            Merge.ix[:, 1] = df
            Merge.columns = [Merge.columns[1]] * 2
            df2 = pd.concat([df2, Merge], axis=1, join="outer")
        return df2.ix[:, np.arange(df2.columns.size) %2 != 0], df2.ix[:, np.arange(df2.columns.size) %2 == 0]
    else:
        period = {'日': 0, '周': 1, '月': 2, '残缺': 2}
        df2 = pd.DataFrame()
        for i, name in enumerate(data.columns):
            df = pd.DataFrame()
            if '日' in p[i]:
                data1 = pd.DataFrame(data[name].dropna())
            elif '周' in p[i]:
                data1 = pd.DataFrame(data[name].dropna())
#                data1 = pd.DataFrame(data.dropna())
            else:
                data1 = pd.DataFrame(data[name].dropna())
#                data1 = pd.DataFrame(data.dropna())
            start_time = pd.concat([benchmark.ix[:, i].dropna(
            ), data1], axis=1, join='inner').index[0].strftime('%Y-%m-%d')
            end_time = data1.index[-1].strftime('%Y-%m-%d')
            part_benchmark = pd.DataFrame(
                benchmark.ix[start_time: end_time, i].dropna())
            Merge = pd.merge(
                left=part_benchmark,
                right=data1,
                how='left',
                left_index=True,
                right_index=True)
            Merge[name].iloc[-1] = data1[name].values[-1]
            for i, j in zip(Merge[name].dropna().index[:-1],
                            Merge[name].dropna().index[1:]):
                df = pd.concat([df, Merge[name][i:j].fillna(
                    (Merge[name][j] / Merge[name][i]) ** (1 / (len(Merge[name][i:j]) - 1)))[:-1].cumprod()])
            df = pd.concat([df, Merge[name][-1:]])
            df.columns = data1.columns
            Merge.ix[:, 1] = df
            Merge.columns = [Merge.columns[1]] * 2
            df2 = pd.concat([df2, Merge], axis=1, join="outer")
        return df2.ix[:, np.arange(df2.columns.size) %2 != 0], df2.ix[:, np.arange(df2.columns.size) %2 == 0]


def interpolate2(data, benchmark, p, strategy=None):
    period = {'日': 0, '周': 1, '月': 2, '残缺': 2}
    calendar = benchmark.iloc[:,0]
    df_l = []
    def interpolate_wrap_day(x):
        x_dropped = pd.DataFrame(x.dropna())
        if len(x_dropped)<3:
            return pd.np.nan
        else:
            start = x_dropped.index[0]
            end = x_dropped.index[-1]
            calendar_tmp = pd.DataFrame(calendar.loc[start:end])
            tmp_df = pd.merge(calendar_tmp,x_dropped,how='left',left_index=True,right_index=True)            
            return tmp_df.iloc[:,1].interpolate()
        
    data1 = data.apply(lambda x:interpolate_wrap_day(x))
    
    if data.columns.size == 1 and strategy is None:
        for i,p1 in enumerate(p):
            benchmark1 = pd.DataFrame(benchmark.iloc[:,period[p1]])
            data_tmp = pd.DataFrame(data1.iloc[:,i])
            df_l.append(pd.merge(left=benchmark1,right=data_tmp,how='left',left_index=True,right_index=True).dropna())
        df2 = pd.concat(df_l,axis=1,join='outer')
        left,right = df2.loc[:, np.arange(df2.columns.size) %2 != 0], df2.loc[:, np.arange(df2.columns.size) %2 == 0]
        
    elif data.columns.size > 1 and strategy is None:
        for i,p1 in enumerate(p):
            benchmark1 = pd.DataFrame(benchmark.iloc[:,period[p1]])
            data_tmp = pd.DataFrame(data1.iloc[:,i])
            df_l.append(pd.merge(left=benchmark1,right=data_tmp,how='left',left_index=True,right_index=True).dropna())
        df2 = pd.concat(df_l,axis=1,join='outer')
        left,right = df2.loc[:, np.arange(df2.columns.size) %2 != 0], df2.loc[:, np.arange(df2.columns.size) %2 == 0]
        right.columns = left.columns
        
    elif data.columns.size == 1 and strategy is not None :
        for i,p1 in enumerate(p):
            benchmark1 = pd.DataFrame(strategy.iloc[:,period[p1]])
            data_tmp = pd.DataFrame(data1.iloc[:,i])
            df_l.append(pd.merge(left=benchmark1,right=data_tmp,how='left',left_index=True,right_index=True).dropna())
        df2 = pd.concat(df_l,axis=1,join='outer')
        left,right = df2.loc[:, np.arange(df2.columns.size) %2 != 0], df2.loc[:, np.arange(df2.columns.size) %2 == 0]
    else:
        for i,p1 in enumerate(p):
            benchmark1 = pd.DataFrame(strategy.iloc[:,i])
            data_tmp = pd.DataFrame(data1.iloc[:,i])
            df_l.append(pd.merge(left=benchmark1,right=data_tmp,how='left',left_index=True,right_index=True).dropna())
        df2 = pd.concat(df_l,axis=1,join='outer')
        left,right = df2.loc[:, np.arange(df2.columns.size) %2 != 0], df2.loc[:, np.arange(df2.columns.size) %2 == 0]
        right.columns = left.columns
    return left,right

def creat_simu_strategy_index(navs, strategy):
    # 从mysql中读取数据
    clzs = get_data_from_mysql('select index_date, \
                                all_market as 全市场策略, \
                                macro as 宏观对冲策略, \
                                relative as 相对价值策略, \
                                cta as 管理期货策略, \
                                stocks as 股票多头策略, \
                                bonds as 固定收益策略  from t_strategy_index', 'index_date')
    clzs.index = pd.to_datetime(clzs.index)
    df = pd.DataFrame()
    for i, sn in enumerate(strategy):  # sn=strategy_name
        sn = strategies_dict[sn]
        nav = navs.ix[:, i].dropna()
        clzs_union = clzs[nav.index[0]:nav.index[-1]][sn]
        clzs_union.name = sn
        clzs_union = pd.DataFrame(clzs_union)
        df = pd.concat([df, clzs_union], axis=1, join='outer')
    return df

def creat_simu_strategy_index2(nav, strategy):
    # 从mysql中读取数据
    clzs = get_data_from_mysql('select index_date, \
                                all_market as 全市场策略, \
                                macro as 宏观对冲策略, \
                                relative as 相对价值策略, \
                                cta as 管理期货策略, \
                                stocks as 股票多头策略, \
                                bonds as 固定收益策略  from t_strategy_index', 'index_date')
    clzs.index = pd.to_datetime(clzs.index)
#    nav=navs_and_date_raw.iloc[:,1]
#    strategy=products_info['strategy'][0]
#    df = pd.DataFrame()
    sn = strategies_dict[strategy]
#    nav = navs.ix[:, i].dropna()
    clzs_union = clzs[nav.index[0]:nav.index[-1]][sn]
    clzs_union.name = sn
    clzs_union = pd.DataFrame(clzs_union)
    return  clzs_union


def creat_strategy_index(navs, strategy):
    #    clzs = pd.read_excel('Strategies_Index\川宝策略指数.xlsx', index_col=0)
    # 从mysql中读取数据
    clzs = get_data_from_mysql('select index_date, \
                                all_market as 全市场策略, \
                                macro as 宏观对冲策略, \
                                relative as 相对价值策略, \
                                cta as 管理期货策略, \
                                stocks as 股票多头策略, \
                                bonds as 债券策略  from t_strategy_index', 'index_date')
    clzs.index = pd.to_datetime(clzs.index)
    df = pd.DataFrame()
    for i, sn in enumerate(strategy):  # sn=strategy_name
        sn = strategies_dict[sn]
        nav = navs.ix[:, i].dropna()
        clzs_union = clzs[nav.index[0]:nav.index[-1]][sn]
#        clzs_union = clzs[navs.ix[:, i].dropna().index[0]:navs.ix[:, i].dropna().index[-1]][sn]
#        clzs_union.name = navs.columns[i] + "-" + sn + "-" + str(i)
        clzs_union.name = sn
        clzs_union = pd.DataFrame(clzs_union)
        df = pd.concat([df, clzs_union], axis=1, join='outer')
    return df

def creat_public_strategy_index(navs, strategy):
    # 从mysql中读取数据
    clzs = get_data_from_mysql('select index_date, \
                                commingled as 混合型, \
                                bonds as 债券型, \
                                money as 货币型, \
                                stocks as 股票型 \
                                from t_public_strategy_index', 'index_date')
    
    clzs.index = pd.to_datetime(clzs.index)
    df = pd.DataFrame()
    for i, sn in enumerate(strategy):  # sn=strategy_name
        sn = strategies_dict[sn]
        nav = navs.ix[:, i].dropna()
        clzs_union = clzs[nav.index[0]:nav.index[-1]][sn]
        clzs_union.name = sn
        clzs_union = pd.DataFrame(clzs_union)
        df = pd.concat([df, clzs_union], axis=1, join='outer')
    return df

def creat_public_strategy_index2(nav, strategy):
    # 从mysql中读取数据
    clzs = get_data_from_mysql('select index_date, \
                                commingled as 混合型, \
                                bonds as 债券型, \
                                money as 货币型, \
                                stocks as 股票型 \
                                from t_public_strategy_index', 'index_date')
   
    clzs.index = pd.to_datetime(clzs.index)
   # sn=strategy_name
    sn = strategies_dict[strategy]
    clzs_union = clzs[nav.index[0]:nav.index[-1]][sn]
    clzs_union.name = sn
    clzs_union = pd.DataFrame(clzs_union)
    return clzs_union


# 年化收益
def annRets(navs):  # for pandas
    l = []
    navs = pd.DataFrame(navs)
    for i in range(navs.columns.size):
        nav = navs.ix[:, i].dropna()
        days = (nav.index[-1] - nav.index[0]).days
        try:
            l.append((nav.iloc[-1] / nav.iloc[0] - 1) * (365 / days))
        except BaseException:
            l.append(np.nan)
    return l


# 年化波动
def annVol(rets, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    rets = pd.DataFrame(rets)
    l = []
    for i in range(rets.columns.size):
        ret = rets.ix[:, i].dropna()
        l.append(ret.std() * np.sqrt(period[p[i]]))
    return l


# 最大回撤
def maxdrawdown(navs):
    navs = pd.DataFrame(navs)
    l = []
    for i in range(navs.columns.size):
        nav = navs.ix[:, i].dropna()
        if len(nav) == 0:
            l.append(np.nan)
        else:
            endtime = np.argmax(np.maximum.accumulate(nav) - nav)
            starttime = np.argmax(nav[:endtime])
            high = nav[starttime]
            low = nav[endtime]
            l.append((low - high) / high * -1)
    return l


# 平均回撤
def meandrawdown(navs):
    l = []
    navs = pd.DataFrame(navs)
    for i in range(navs.columns.size):
        nav = navs.ix[:, i].dropna()
        drawdown = (nav - np.maximum.accumulate(nav)) / \
            np.maximum.accumulate(nav)
        l.append(drawdown.mean() * -1)
    return l


# 夏普
def SharpRatio(rets, p):
    # for pandas
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    rets = pd.DataFrame(rets)
    l = []
    for i in range(rets.columns.size):
        ret = rets.ix[:, i].dropna()
        try:
            l.append((ret -
                      return_noRisk /
                      period[p[i]]).mean() /
                     ret.std() *
                     np.sqrt(period[p[i]]))
        except BaseException:
            l.append(9999)
    return l


# calmar
def Calmar(rets, maxdown):
    # for pandas
    l = []
    if isinstance(rets,list):
        pass
    else:
        rets = pd.DataFrame(rets)
    try:
        NO = rets.columns.size
        for i in range(NO):
            if maxdown[i] == 0:
                l.append(9999)
            else:
                l.append(rets.ix[:, i].mean() / maxdown[i])
    except BaseException:
        NO = len(rets)
        for i in range(NO):
            if maxdown[i] == 0 or maxdown[i] == '':
                l.append(9999)
            else:
                l.append(rets[i] / maxdown[i])
    return l


# 索提诺
def Sortino(rets, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    l = []
    rets = pd.DataFrame(rets)
    try:
        NO = rets.columns.size
        for i in range(NO):
            return_noRisk_new = return_noRisk / period[p[i]]
            l.append((rets.ix[:, i] -
                      return_noRisk /
                      period[p[i]]).mean() /
                     downsideRisk2(rets.ix[:, i], return_noRisk_new) * np.sqrt(period[p[i]]))
    except BaseException:
        NO = len(rets)
        for i in range(NO):
            return_noRisk_new = return_noRisk / period[p[i]]
            l.append((rets[i] - return_noRisk / period[p[i]]
                      ).mean() / downsideRisk2(rets[i], return_noRisk_new))
    return l




def Stutzer2(rets, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    l = []
    rets = pd.DataFrame(rets)
    for i, name in enumerate(rets.columns):
        try:
            ret = rets[name].dropna()
            rt = ret - return_noRisk / period[p[i]]

            def func(theta, ret): return np.log(
                (np.e ** (theta * ret)).sum() / len(ret))
            max_theta = sco.minimize(
                func, (-25.,), method='SLSQP', bounds=((-50, 5),), args=(rt,)).x
            lp = func(max_theta, rt)
            stutzer_index = (np.sign(rt.mean()) *
                             np.sqrt(2 * abs(lp) * period[p[i]]))
            l.append(stutzer_index)
        except BaseException:
            l.append(9999)
    return l  # ,max_theta


def Er_and_Stu_bound(rets, cons, p, call='Er'):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    ann = period[p[0]]
    if call == 'Stu':
        R_f = return_noRisk / ann
        return (rets.mean() - R_f - cons * (rets.std() /
                                            np.sqrt(rets.size))) / (rets.mean() - R_f)
    elif call == 'Er':
        return (rets.mean() - cons * (rets.std() /
                                      np.sqrt(rets.size))) * (365 / ann)
    else:
        return 0


def VARRatio(c, rets,p):
    # for pandas
    period = {'日': 1, '周': 5, '月': 21}
    rets = pd.DataFrame(rets)
    l = []
    for i in range(rets.columns.size):
        ret = rets.ix[:,i].dropna()
        mu = ret.mean()
        sigma = ret.std()
        alpha = stats.norm.ppf(1 - c, mu, sigma)
        var_day=alpha/period[p[i]]
        Var=pd.Series(var_day)
        Var= Var.agg([lambda x,i=i: x * np.sqrt(i) for i in [10]]).T
        Var=(Var.values)[0][0]
        l.append(Var)
    return l


def CVar(profits, c,p):
    period = {'日': 1, '周': 5, '月': 21}
    profits = pd.DataFrame(profits)
    l = []
    for i in range(profits.columns.size):
        mu = profits.ix[:,i].mean()
        sigma = profits.ix[:, i].std()
        cvar = - 1 + np.exp(mu) * stats.norm.cdf(-sigma -
                                                 stats.norm.ppf(c)) / (1 - c)
        cvar_day=cvar/period[p[i]]
        CVar=pd.Series(cvar_day)
        CVar= CVar.agg([lambda x,i=i: x * np.sqrt(i) for i in [10]]).T
        CVar=(CVar.values)[0][0]
        l.append(CVar)
    return l


# Alpha&Beta
def alpha_and_beta(rets, i_rets,p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    l_alpha = []
    l_beta = []
    l_r2 = []
    rets = pd.DataFrame(rets)
    i_rets = pd.DataFrame(i_rets)
    i_rets.columns = rets.columns

    for i, name in enumerate(rets.columns):
        if len(rets) != 0:
            ret = rets[name].dropna() - return_noRisk/period[p[i]]
            i_ret = i_rets[name].dropna() - return_noRisk/period[p[i]]
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                i_ret, ret)
            l_alpha.append(intercept)
            l_beta.append(slope)
            l_r2.append(r_value ** 2)
        else:
            l_alpha.append(np.nan)
            l_beta.append(np.nan)
            l_r2.append(np.nan)
    return l_alpha, l_beta, l_r2


# 特雷诺
def TreynorRatio(navs, beta, p=None, products_info=None):
    l = []
    try:
        NO = navs.columns.size
        for i in range(NO):
            l.append((annRets(navs.ix[:, i])[0] - return_noRisk) / beta[i])
    except BaseException:
        NO = len(navs)
        for i in range(NO):
            l.append((annRets(navs[i])[0] - return_noRisk) / beta[i])
    return l

def Burke_ratio(navs_and_date_raw):
    l=[]
    a,b= signal_drawdown(navs_and_date_raw)
    a=pd.DataFrame(a)
    for i in range(a.columns.size):
        yearret=annRets(navs_and_date_raw)[i]
        Signal_drawdown=a.iloc[:,i].dropna()
        if Signal_drawdown.index.size !=0 and ((Signal_drawdown*Signal_drawdown).sum()/Signal_drawdown.index.size)!=0 :
             burke=( yearret-return_noRisk)/(np.sqrt((Signal_drawdown*Signal_drawdown).sum()/Signal_drawdown.index.size))
             l.append(burke)
        else:
             l.append(9999)
    return l  
    
def Sterling(navs_and_date_raw):
    l=[]
    a,b= signal_drawdown(navs_and_date_raw)
    a=pd.DataFrame(a)
    for i in range(a.columns.size):
        yearret=annRets(navs_and_date_raw)[i]
        Signal_drawdown=a.iloc[:,i].dropna()
        five_signal_drawdown= Signal_drawdown.sort_values(ascending=False).head(5) 
        if ((five_signal_drawdown* five_signal_drawdown).sum()/5)!=0 :
            sterling=(yearret-return_noRisk)/(np.sqrt((five_signal_drawdown* five_signal_drawdown).sum()/5))
            l.append(sterling)
        else:
             l.append(9999)
    return l    
# 信息比率
#def InfoRation(rets, index, p):
#    # for pandas
#    rets = pd.DataFrame(rets)
#    index = pd.DataFrame(index)
#    l = []
#    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
#    for i, name in enumerate(rets.columns):
#        TE = rets[name] - index[name]
#        TE2 = TE.std()
#        l.append(TE.mean() * np.sqrt(period[p[i]]) / TE2)
#    return l
def InfoRation(rets, index_rets, p):
    # for pandas
    rets = pd.DataFrame(rets)
    index_rets= pd.DataFrame(index_rets)
    l = []
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    for i in range(rets.columns.size):  
        ret=rets.iloc[:,i].dropna().mean()
        if len(index_rets.columns)==1:
            index_ret=index_rets.iloc[:,0].dropna().mean()  
        elif len(index_rets.columns)>1:
            index_ret=index_rets.iloc[:,i].dropna().mean() 
        TE = ret - index_ret
        TE2 = rets.iloc[:,i].dropna().std()
        l.append(TE * np.sqrt(period[p[i]]) / TE2)
    return l
    

def MCV(rets, i_rets, alpha, beta, p):
    l = []
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    return_noRisk_new = []
    for i in range(rets.columns.size):
        return_noRisk_new.append(return_noRisk / period[p[i]])
    for j in range(rets.columns.size):
        l.append(alpha[j] /
                 ((rets.std().tolist()[j] /
                   i_rets.std().tolist()[j] -
                     beta[j]) *
                  abs(i_rets.mean().tolist()[j] -
                      return_noRisk_new[j])))
    return l


def Continuous(rets):
    l = []
    rets=pd.DataFrame(rets)
    for name in rets.columns:
        ret = rets[name].dropna()
        x = ret.iloc[:-1]
        y = ret.iloc[1:]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        t_test = stats.ttest_ind(x, y)
        l.append((slope, t_test[1]))
    return l


def UpDownRatio(rets):
    l = []
    for name in rets.columns:
        t1 = rets[name][rets[name] > 0].mean()
        l.append(t1 / abs(rets[name][rets[name] <= 0].mean() + 1e-4))
    return l


def WinRate(rets):
    l = []
    for name in rets.columns:
        l.append(float((rets[name] > 0).sum() / rets[name].size))
    return l


def WinRate2(rets, i_rets):
    l = []
    rets = pd.DataFrame(rets)
    i_rets = pd.DataFrame(i_rets)
    for i, name in enumerate(rets.columns):
        ret = rets[name].dropna()
        i_ret = i_rets.ix[:, i].dropna()
        l.append(float((ret > i_ret).sum() / ret.size))
    return l
# ==============================================================================
# TM,HM,CL
# ==============================================================================
# 默认profit与hs300profit长度相同，对齐
# 截距越大代表基金经理有选股能力，一次项系数代表组合承担的系统风险，二次项系数越大代表基金经理有择时能力
# 截距、一次项系数、二次项系数


def TM(profit, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    return_noRisk_new = return_noRisk / period[p[0]]
    y = profit - return_noRisk_new
    x = pd.concat([hs300Profit - return_noRisk_new,
                   (hs300Profit - return_noRisk_new) ** 2],
                  axis=1,
                  join='outer')
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    t_test = stats.ttest_ind(x, y)
    return [regr.intercept_, regr.coef_[0][0], regr.coef_[0][1],t_test]


# 默认profit与hs300profit长度相同，对齐
# 截距越大代表基金经理有选股能力，一次项系数+二次项系数越大代表基金经理有择时能力
# 截距、一次项系数、二次项系数
def HM(profit, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    return_noRisk_new = return_noRisk / period[p[0]]
    noRisk_profit = profit - return_noRisk_new
    noRisk_index = hs300Profit - return_noRisk_new
    y = noRisk_profit
    x = pd.concat([noRisk_profit, noRisk_index.where(
        noRisk_index > 0, 0)], axis=1, join='outer')
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    t_test = stats.ttest_ind(x, y)
    [regr.intercept_, regr.coef_[0][0], regr.coef_[0][1],t_test]
    return [regr.intercept_, regr.coef_[0][0], regr.coef_[0][1],t_test]


# 默认profit与hs300profit长度相同，对齐
# 截距越大代表基金经理有选股能力，一次项系数-二次项系数越大代表基金经理有择时能力
# 截距、一次项系数、二次项系数
def CL(rets, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    l = []
    for i, name in enumerate(rets):
        return_noRisk_new = return_noRisk / period[p[i]]
        noRisk_profit = rets[name].dropna() - return_noRisk_new
        noRisk_index = hs300Profit.iloc[:, i].dropna() - return_noRisk_new
        y = noRisk_profit
        x = pd.concat([noRisk_profit.where(noRisk_profit < 0, 0), noRisk_index.where(
            noRisk_index > 0, 0)], axis=1, join='outer')
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        t_test = stats.ttest_ind(x, y)
        l.append([regr.intercept_, regr.coef_[1], regr.coef_[0],t_test])
    return l

#以下为张焕芳添加，主要添加了M2,omega,单一最大回撤，以及
#区间段累计收益率
def to_today_ret(navs):
    l=[]
    navs=pd.DataFrame(navs)
    for i in range(navs.columns.size):
        nav=navs.iloc[:,i].dropna()
        if nav.size==0:
           ret= "--"
        else:
           ret=(nav[-1]-nav[0])/nav[0]
        l.append(ret) 
    return l 
#情景分析
def scene_analyse(navs):    
    # ==============================================================================
        #      情景分析
    #========#==============================================================================    
   #股灾
#    stock_crics=['2015-06-15','2015-08-28']#2015年股灾
#    stock_hot=['2014-11-20','2014-12-22']#2014年股票风格转换
#    bank_i=['2013-05-09','2013-06-20']#2013年中钱荒
#    bond_crics=['2016-11-24','2016-12-20']#2015年债市风暴
#    fusing=['2016-01-01','2016-01-07']#2016年初熔断
    #cta=['2016-04-21','2016-05-24']商品大反转
    #市场指数的在这段时间段的净值 股灾
    hs300_all= pd.read_sql('select index_date,hs300 as 沪深300 from t_wind_index_market',con,index_col='index_date')
    hs300_day=hs300_all.iloc[:,0]
    hs300_stock_crics=hs300_day['2015-06-15':'2015-08-28']
    hs300_stock_hot=hs300_day['2014-11-20':'2014-12-22']
    hs300_stock_crics_ret=to_today_ret(hs300_stock_crics)#沪深300在股灾的损失
    hs300_stock_hot_ret=to_today_ret( hs300_stock_hot)#沪深300在股市的盈利
    hs300_fusing=hs300_day['2016-01-01':'2016-01-07']
    hs300_fusing_ret=to_today_ret(hs300_fusing)#沪深300在熔断的亏损
    
    
    bond_index = pd.read_sql('select index_date,csi_total_debt as 中证全债指数 from t_wind_index_market',con,index_col='index_date')
    nh_index= pd.read_sql('select index_date,south_china_commodity as 南华商品指数 from t_wind_index_market',con,index_col='index_date')
#    zz500= pd.read_sql('select index_date,zz500 as 中证500 from t_wind_index_market',con,index_col='index_date')
    bond_index_crics= bond_index['2016-11-24':'2016-12-20']
    bond_index_crics_ret=to_today_ret( bond_index_crics)
    bond_index_bank_i=bond_index['2013-05-09':'2013-06-20']
    bond_index_bank_i_ret=to_today_ret(bond_index_bank_i)
    
    nh_index_cta= nh_index['2016-04-21':'2016-05-24']#南华商品指数在商品大反转的盈利
    nh_index_cta_ret=to_today_ret(nh_index_cta)
   
    nh=['--','--','--', nh_index_cta_ret[0],'--','--']
    bond=[bond_index_bank_i_ret[0],'--','--','--', bond_index_crics_ret[0],'--']
    hs300=['--',hs300_stock_hot_ret[0],hs300_stock_crics_ret[0], nh_index_cta_ret[0],'--', hs300_fusing_ret[0]]
    index_rets=pd.concat([pd.DataFrame(hs300), pd.DataFrame(nh),pd.DataFrame(bond)],axis=1)
    index_rets.columns=['沪深300','南华商品指数','中证全债']
    index_rets=index_rets.T
    index_rets.columns=['2013年中钱荒','2014年股票风格转换','2015年股灾','商品大反转','2015年债市风暴','2016年初熔断']
   #产品的情景分析
    navs_date_stock_crics=navs['2015-06-15':'2015-08-28']
    navs_date_stock_hot=navs['2014-11-20':'2014-12-22']
    navs_date_bank_i=navs['2013-05-09':'2013-06-20']
    navs_date_cta=navs['2016-04-21':'2016-05-24']
    navs_date_bond_crics=navs['2016-11-24':'2016-12-20']
    navs_date_fusing=navs['2016-01-01':'2016-01-07']
     
   #在这段时间的收益情况
    profits_stock_crics=  to_today_ret(navs_date_stock_crics)
    profits_stock_hot=  to_today_ret(navs_date_stock_hot)
    profits_bank_i=  to_today_ret(navs_date_bank_i)
    profits_bond_crics=  to_today_ret(navs_date_bond_crics)
    profits_bond_fusing=  to_today_ret(navs_date_fusing)
    profits_cta= to_today_ret( navs_date_cta)
    things_profits= pd.concat([pd.DataFrame( profits_bank_i), pd.DataFrame(profits_stock_hot),pd.DataFrame(profits_stock_crics),pd.DataFrame(profits_cta),\
             pd.DataFrame(profits_bond_fusing), pd.DataFrame( profits_bond_crics)],axis=1)
   
    things_profits.index=navs.columns
    things_profits.columns=['2013年中钱荒','2014年股票风格转换','2015年股灾','商品大反转','2015年债市风暴','2016年初熔断']
    scene_analysis=pd.concat([things_profits,index_rets])       
    return  scene_analysis
    
def Omega(rets,p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    rets=pd.DataFrame(rets)
    return_noRisk_new = return_noRisk / period[p[0]]
    l=[]
    for i in range(rets.columns.size):   
        more_ret=rets.ix[:,i]- return_noRisk_new
        profit_ret=more_ret[more_ret>0].sum()
        loss_ret=abs(more_ret[more_ret<0].sum())
        omega= profit_ret/loss_ret
        l.append(omega)
    return l   


#M平方测度  张焕芳加
def M2(navs, hs300_ret, p):
    l = []
    navs=pd.DataFrame(navs)#数据库产品适用性待核查
    try:
        NO = navs.columns.size
        for i in range(NO):
            l.append((annRets(navs.ix[:, i])[0] - return_noRisk) * annVol(hs300_ret,p)[0]/annVol(navs.ix[:, i].pct_change(), '日')[0])
    except BaseException:
        NO = len(navs)
        for i in range(NO):
            l.append((annRets(navs[i])[0] - return_noRisk) * annVol(hs300_ret, '日')[0]/annVol(navs.ix[:, i].pct_change(), '日')[0])
    return l


 #最大单一回撤张焕芳添加
def signal_drawdown(navs):
    signal_drawdowns=pd.DataFrame()
    l=[]
    navs=pd.DataFrame(navs)
    for i in range(navs.columns.size): 
        nav=navs.iloc[:,i].dropna()
        ret=(nav-nav.shift(1))
        up_down=ret[ret!=0]
        up_down1=up_down>0
        jizhi=(up_down1+up_down1.shift(-1))==1
        jizhi.ix[0,0]=1#为了不舍弃净值首末值，即将首末值默认为极值
        jizhi.ix[-1,-1]=1
        jizhi_data=jizhi[jizhi==1].index.tolist()
        nav_jizhi=nav.loc[jizhi_data]
        ret_jizhi=nav_jizhi.pct_change()
        signal_drawdown=ret_jizhi[ret_jizhi<0]*-1#连续单一回撤
        signal_maxdrawdown= signal_drawdown.max()#最大单一回撤
        signal_drawdowns=pd.concat([signal_drawdowns,signal_drawdown],axis=1)
        l.append(signal_maxdrawdown)
    return signal_drawdowns, l

#to json 文件，张焕芳添加
def dataFrameToJson(df):
    df = pd.DataFrame(df)
    df = df.replace(np.nan,'--')
    colName = list(df.columns)
    dfIndex= list(df.index)
    dfValue = []
    for i in colName:
        dfValue.append(list(df[i]))
    value = []
    for i in range(len(colName)):
        try:
            aa = {'name': colName[i], 'data': list(map(float,dfValue[i]))}
            value.append(aa)
        except:
            aa = {'name': colName[i], 'data': dfValue[i]}
            value.append(aa)
    dfIndex = list(map(str,dfIndex))
    result = json.dumps({'参数':value,'指数':dfIndex})
    return result

def dataFrameToJson_table(df,label1,nobel,k):
    df = pd.DataFrame(df)
    df = df.replace(np.nan,'--')
    colName = list(df.columns)
    dfIndex= list(df.index)
    dfValue = []
    for i in colName:
        dfValue.append(list(df[i]))
    value = []
    for i in range(len(colName)):
        try:
            aa = {'name': colName[i], 'data': list(map(float,dfValue[i]))}
            value.append(aa)
        except:
            aa = {'name': colName[i], 'data': dfValue[i]}
            value.append(aa)
    dfIndex = list(map(str,dfIndex))
    result = json.dumps({'参数':value,'指数':dfIndex,'小标题':[label1],'指标解释':[nobel],'flag':[k]})
    return result
def dataFrameToJson_table2(df,label1,label2,nobel):#针对有三级标题的
    df = pd.DataFrame(df)
    df = df.replace(np.nan,'--')
    colName = list(df.columns)
    dfIndex= list(df.index)
    dfValue = []
    for i in colName:
        dfValue.append(list(df[i]))
    value = []
    for i in range(len(colName)):
        try:
            aa = {'name': colName[i], 'data': list(map(float,dfValue[i]))}
            value.append(aa)
        except:
            aa = {'name': colName[i], 'data': dfValue[i]}
            value.append(aa)
    dfIndex = list(map(str,dfIndex))
    result = json.dumps({'参数':value,'指数':dfIndex,'小标题':[label1],'三级标题':[label2],'指标解释':[nobel]})
    return result
def dataFrameToJson_figure(df,label1,nobel):
    df = pd.DataFrame(df)
    df = df.replace(np.nan,'--')
    colName = list(df.columns)
    dfIndex= list(df.index)
    dfValue = []
    for i in colName:
        dfValue.append(list(df[i]))
    value = []
    for i in range(len(colName)):
        try:
            aa = {'name': colName[i], 'data': list(map(float,dfValue[i]))}
            value.append(aa)
        except:
            aa = {'name': colName[i], 'data': dfValue[i]}
            value.append(aa)
    dfIndex = list(map(str,dfIndex))
    result = json.dumps({'参数':value,'指数':dfIndex,'小标题':[label1],'图形解释':nobel})
    return result
#扫描统计量 张焕芳添加#扫描统计量 张焕芳添加
def saomiao2(rets,p):
        l=[]
        rets=pd.DataFrame(rets)
        period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
        for i in range(rets.columns.size): 
            return_noRisk_new = return_noRisk / period[p[0]]
            ret=rets.iloc[:,i].dropna()
            win_bool=( ret- return_noRisk_new)>0
        #判断赢的最长连
            k=[]
            length=0
            for j in range(win_bool.size):  
                if win_bool[j]==1:   
                    length += 1
                else:
                    length=0
                k.append(length)        
            m=max(k)
            if m==0:
                
                l.append(0.3)
            elif m==ret.size or m==ret.size-1 or m==ret.size-2: 
                l.append(1)
            else: 
                p_win=0.3
                for i in range(70):
                    q=1-p_win
                    Q2=1-p_win**m*(1+m*q)
                    Q3=1-p_win**m*(1+2*m*q)+0.5*p_win**(2*m)*(2*m*q+m*(m-1)*q*q)
                    P=1-Q2*(Q3/Q2)**(ret.size/m-2)
                    if P>=0.95:
                        l.append(p_win)
                        break
                    else:
                        p_win += 0.01
        return l
#HM,CL,TM 张焕芳添加了t检验输出参数
def TM2(profit, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    return_noRisk_new = return_noRisk / period[p[0]]
    y = profit - return_noRisk_new
    x = pd.concat([hs300Profit- return_noRisk_new,
                   (hs300Profit - return_noRisk_new) ** 2],
                  axis=1,
                  join='outer')
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    t_test = stats.ttest_ind(x, y)
    return [regr.intercept_, regr.coef_[0][0], regr.coef_[0][1],t_test]


# 默认profit与hs300profit长度相同，对齐
# 截距越大代表基金经理有选股能力，一次项系数+二次项系数越大代表基金经理有择时能力
# 截距、一次项系数、二次项系数
def HM2(profit, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    return_noRisk_new = return_noRisk / period[p[0]]
    noRisk_profit = profit - return_noRisk_new
    noRisk_index = hs300Profit - return_noRisk_new
    y = noRisk_profit
    x = pd.concat([noRisk_profit, noRisk_index.where(
        noRisk_index > 0, 0)], axis=1, join='outer')
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    t_test = stats.ttest_ind(x, y)
    return [regr.intercept_, regr.coef_[0][0], regr.coef_[0][1],t_test]


# 默认profit与hs300profit长度相同，对齐
# 截距越大代表基金经理有选股能力，一次项系数-二次项系数越大代表基金经理有择时能力
# 截距、一次项系数、二次项系数
def CL2(rets, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    l = []
    for i, name in enumerate(rets):
        return_noRisk_new = return_noRisk / period[p[i]]
        noRisk_profit = rets[name].dropna() - return_noRisk_new
        noRisk_index = hs300Profit.iloc[:, i].dropna() - return_noRisk_new
        y = noRisk_profit
        x = pd.concat([noRisk_profit.where(noRisk_profit < 0, 0), noRisk_index.where(
            noRisk_index > 0, 0)], axis=1, join='outer')
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        t_test = stats.ttest_ind(x, y)
        l.append([regr.intercept_, regr.coef_[1], regr.coef_[0],t_test])
    return l
#不同市场环境下的表现
#def differ_market(rets, hs300):  # 传入两个带时间索引的数据框,已经对齐
#        df = pd.concat([rets, hs300], axis=1)
#        df.columns = ['产品', '指数']
#
#        '''牛市回归'''
#        df_niu = df[df['指数'] > 0]
#        y_N, x_N = [df_niu.ix[:, i] for i in df_niu.columns]
#        x_N = sm.add_constant(x_N)
#        Nshi = sm.OLS(y_N, x_N).fit()
#        '''熊市回归'''
#        df_xiong = df[df['指数'] <= 0]
#        y_X, x_X = [df_xiong.ix[:, i] for i in df_xiong.columns]
#        x_X = sm.add_constant(x_X)
#        Xshi = sm.OLS(y_X, x_X).fit()
##        print('牛市：y={:.5f}{:+.5f}X'.format(Nshi.params[0],Nshi.params[1]))
##        print('熊市：y={:.5f}{:+.5f}X'.format(Xshi.params[0],Xshi.params[1]))
##        return df_niu,df_xiong,Nshi.pvalues.values[-1],Xshi.pvalues.values[-1]
#        return df_niu, df_xiong, Nshi, Xshi

def differ_market(rets,i_rets): 
    # 传入两个带时间索引的数据框,已经对齐
      l=[]
      beta_niu=[]
      beta_xiong=[]
      for i in range(rets.columns.size):
          ret=rets.iloc[:,i].dropna()
          i_ret=i_rets.iloc[:,i].dropna()
          df = pd.concat([ret, i_ret], axis=1)
          df.columns = ['产品', '指数']

          '''牛市回归'''
          df_niu = df[df['指数'] > 0]
          y_N, x_N = [df_niu.ix[:, i] for i in df_niu.columns]
          x_N = sm.add_constant(x_N)  
          try:
            Nshi = sm.OLS(y_N, x_N).fit()  
            regression_niu= Nshi.params[1]
          except BaseException:  
            regression_niu= 0
          '''熊市回归'''
          df_xiong = df[df['指数'] <= 0]
          y_X, x_X = [df_xiong.ix[:, i] for i in df_xiong.columns]
          x_X = sm.add_constant(x_X)
      
        
#        if type(Xshi) == int:
#            beta_ratio = Nshi.params[1] - Xshi
#        elif type(Nshi) == int:
#           beta_ratio = Nshi - Xshi.params[1]
#        else:
#           beta_ratio = Nshi.params[1] - Xshi.params[1]
#        
          try:
             Xshi = sm.OLS(y_X, x_X).fit() 
             regression_xiong= Xshi.params[1]
          except BaseException: 
             regression_xiong=0 
#        print('牛市：y={:.5f}{:+.5f}X'.format(Nshi.params[0],Nshi.params[1]))
#        print('熊市：y={:.5f}{:+.5f}X'.format(Xshi.params[0],Xshi.params[1]))
#        return df_niu,df_xiong,Nshi.pvalues.values[-1],Xshi.pvalues.values[-1]
          beta_niu.append(regression_niu)
          beta_xiong.append(regression_xiong)
          l.append(regression_niu-regression_xiong)
      return l,beta_niu,beta_xiong

def fama(nav):
       navs = nav.pct_change().groupby(pd.TimeGrouper('M')).sum().to_period('M')        
       FAMA_FACTOR = pd.read_sql('t_wind_fama_factor',con,index_col='DATE')
       FAMA_FACTOR = FAMA_FACTOR.to_period('M')        
       regr = linear_model.LinearRegression()
       regr.fit(FAMA_FACTOR.loc[navs.index].fillna(0),navs)
       alpha=regr.intercept_*12
#       result = FAMA_FACTOR.loc[navs.index] * regr.coef_
#       result['残差']=navs- result.sum(axis=1)
#       result.columns = ['市场因子(%)','估值因子(%)','盈利因子(%)','投资因子(%)','规模因子(%)','残差(%)']
#       result.index.name = '月份'
#        result.to_html(border=0,formatters={'市场因子':lambda x:'{:.2%}'.format(x),'估值因子':lambda x:'{:.2%}'.format(x),'盈利因子':lambda x:'{:.2%}'.format(x),'投资因子':lambda x:'{:.2%}'.format(x),'规模因子':lambda x:'{:.2%}'.format(x),})
       return alpha 
def fama2(navs):
    navs=pd.DataFrame(navs)
    l=[]
    for i in range (navs.columns.size):
        nav=navs.iloc[:,i].dropna()    
        profit = nav.pct_change().groupby(pd.TimeGrouper('M')).sum().to_period('M')        
        FAMA_FACTOR = pd.read_sql('t_wind_fama_factor',con,index_col='DATE')
        FAMA_FACTOR = FAMA_FACTOR.to_period('M')        
        regr = linear_model.LinearRegression()
        regr.fit(FAMA_FACTOR.loc[navs.index],profit)
        alpha=regr.intercept_*12
        l.append(alpha)
#       result = FAMA_FACTOR.loc[navs.index] * regr.coef_
#       result['残差']=navs- result.sum(axis=1)
#       result.columns = ['市场因子(%)','估值因子(%)','盈利因子(%)','投资因子(%)','规模因子(%)','残差(%)']
#       result.index.name = '月份'
#        result.to_html(border=0,formatters={'市场因子':lambda x:'{:.2%}'.format(x),'估值因子':lambda x:'{:.2%}'.format(x),'盈利因子':lambda x:'{:.2%}'.format(x),'投资因子':lambda x:'{:.2%}'.format(x),'规模因子':lambda x:'{:.2%}'.format(x),})
    return l 
      
def InfoRation2(rets, index, p):
        # for pandas
        l = []
        period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
        for i, name in enumerate(rets.columns):
            TE = rets.iloc[:,i] - index.iloc[:,i]
            TE2 = TE.std()
            l.append(TE.mean() * np.sqrt(period[p[i]]) / TE2)
        return l

#此函数主要计算股票策略以及alpha策略归因
def calc_data(navs,p):
        indicatrix_data = {}
        period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
        hs300 =fetch_index(con)
#    p_judge= {'日': 0, '周': 1, '月': 1, '残缺': 0}
#    hs300 = pd.DataFrame(hs300.iloc[:, p_judge[p[0]]])
        hs300 = pd.DataFrame(hs300.iloc[:, 0])
        hs300.rename(columns={'沪深300-日':'沪深300'},inplace=True)
        indicatrix_data['market300'] = hs300
        profits=calculate_profits(navs)
#        navs=combin
        indicatrix_data['the_navs'], indicatrix_data['300index_navs'] =interpolate(
            pd.DataFrame(navs), pd.DataFrame(indicatrix_data['market300']), p)
        indicatrix_data['300index_pct'] =calculate_profits(
            indicatrix_data['300index_navs'])
        profits=calculate_profits(navs)
        indicatrix_data['the_pct']=calculate_profits(indicatrix_data['the_navs'])
        indicatrix_data['data_annRets'] =annRets(navs)
        indicatrix_data['data_annVol'] =annVol(profits, p)
        indicatrix_data['data_maxdrawdown'] = maxdrawdown(navs)
        indicatrix_data['data_mean_drawdown'] =meandrawdown(navs)
        indicatrix_data['data_Sharp'] =SharpRatio(profits, p)
        indicatrix_data['data_Calmar'] =Calmar(indicatrix_data['data_annRets'], indicatrix_data['data_maxdrawdown'])
        indicatrix_data['data_inforatio']=InfoRation2(indicatrix_data['the_pct'],indicatrix_data['300index_pct'],['日'])
        
        indicatrix_data['data_Sortino'] =Sortino(profits, p)
        indicatrix_data['data_Stutzer'] =Stutzer2(profits, p)
        indicatrix_data['data_var95'] =VARRatio(0.95, profits)
#        indicatrix_data['data_var99'] = functions.VARRatio(0.99, profits)
        indicatrix_data['data_cvar95'] =CVar(profits, 0.95)
        indicatrix_data['data_alpha_famma']=fama(navs)
        
        #下行风险
        indicatrix_data['downrisk'] =downsideRisk2(indicatrix_data['the_pct'], return_noRisk/period[p[0]])*np.sqrt(period[p[0]])
        
#        indicatrix_data['data_cvar99'] = functions.CVar(profits, 0.99)
        indicatrix_data['data_saomiao']=saomiao2(profits,p)
        indicatrix_data['data_WinRate'] =WinRate(profits)
        _,indicatrix_data['data_signaldrawdown']=signal_drawdown(navs)
        indicatrix_data['omega'] = Omega(profits,p)
        indicatrix_data['M2'] =M2(indicatrix_data['the_navs'],indicatrix_data['300index_pct'], p)
        
        indicatrix_data['data_alpha'], indicatrix_data['data_beta'], indicatrix_data['data_R2'] =alpha_and_beta(
            indicatrix_data['the_pct'], indicatrix_data['300index_pct'],p)
        
        
        [HM_alpha,HM_beta1,HM_beta2,HM_p_value]=HM2(pd.DataFrame( indicatrix_data['the_pct']), pd.DataFrame(indicatrix_data['300index_pct']),p)
        CL= CL2(pd.DataFrame( indicatrix_data['the_pct']), pd.DataFrame(indicatrix_data['300index_pct']), p)
        [TM_alpha,TM_beta1,TM_beta2,TM_t_test]= TM2(pd.DataFrame( indicatrix_data['the_pct']), pd.DataFrame(indicatrix_data['300index_pct']), p)

        indicatrix_data['data_HM_alpha']=(1+HM_alpha[0])**period[p[0]]-1
        indicatrix_data['data_TM_alpha']=(1+TM_alpha[0])**period[p[0]]-1
        indicatrix_data['data_CL_alpha']= (1+CL[0][0])**period[p[0]]-1
        indicatrix_data['CL_alpha_p_value']= CL[0][3][1][0]
        indicatrix_data['CL_beta_p_value']=CL[0][3][1][1]
        indicatrix_data['HM_alpha_p_value']=HM_p_value[1][0]
        indicatrix_data['HM_beta_p_value']=HM_p_value[1][1]
        indicatrix_data['TM_alpha_p_valu']=TM_t_test[1][0]
        indicatrix_data['TM_beta_p_value']=TM_t_test[1][1]
        indicatrix_data['CL_beta_ratio']= CL[0][2]- CL[0][1]
        indicatrix_data['HM_beta2'] =HM_beta2
        indicatrix_data['TM_beta2'] =TM_beta2
        df_niu2, df_xiong2, regression_niu, regression_xiong = differ_market(indicatrix_data['the_pct'], indicatrix_data['300index_pct'])
        indicatrix_data['niu_beta']=regression_niu.params[1]
        indicatrix_data['xiong_beta']=regression_xiong.params[1]
        #贝塔择时比率
        indicatrix_data['beta_zeshi']= indicatrix_data['niu_beta']-indicatrix_data['xiong_beta']
    
        return  indicatrix_data
#以下为陈凯编部分
def creat_index(navs, hs300):
    index = pd.DataFrame()
    for i in range(navs.columns.size):
        tmp = pd.concat([hs300, navs.ix[:, i]], axis=1,
                        join='inner').dropna()[hs300.name]
        index = pd.concat([index, tmp], axis=1, join='outer')
    index.columns = navs.columns
    return index

# ==============================================================================
# 净值走势图
# ==============================================================================

def areaChart_Equity(
        navs_and_date_raw,
        products_info,
        strategy=None,
        market=None):
    if navs_and_date_raw.columns.size == 1:
        s4 = "\nvar d_areaChart_Equity=["

        s4 += "\n{\nname: '" + products_info['product_name'][0] + "',\ndata:"
        s4 += navs_and_date_raw.reset_index().to_json(orient='values', double_precision=4)
        s4 += "\n},"
        
        s4 += "\n{\nname: '" + \
            strategies_dict[products_info['strategy'][0]] + "',\ndata:"
        s4 += strategy.reset_index().to_json(orient='values', double_precision=4)
        s4 += "\n},"

        s4 += "\n{\nname: '" + market.name + "',\ndata:"
        s4 += market[navs_and_date_raw.index[0]:navs_and_date_raw.index[-1]
                     ].reset_index().to_json(orient='values', double_precision=4)
        s4 += "\n},"
        
        s4 += "\n]"
    else:
        s4 = "\nvar d_areaChart_Equity_series=["
        for name in navs_and_date_raw.columns:
            s4 += "\n{\nname: '" + str(name) + "',\ndata:"
            s4 += navs_and_date_raw[name].dropna().reset_index(
            ).to_json(orient='values', double_precision=4)
            s4 += "\n},"
        s4 += "\n]\n"
    return s4



# ==============================================================================
# 直方图
# ==============================================================================
def histChart_DailyRetun(profits):
    def new_two_col(profits, name=profits.columns[0]):
        tmp = pd.DataFrame()
        tmp['rets100'] = profits[name].dropna() * 100
        tmp['hist'] = np.arange(len(tmp['rets100']))
        return tmp

    if profits.columns.size == 1:
        s6 = "\nvar d_dis_rtn ="
        tmp = new_two_col(profits)
        s6 += tmp['rets100'].to_json(orient='values', double_precision=4)
        s6 += "\n"
    else:
        s6 = "\nvar d_histChart_DailyReturn_series =["
        for name in profits.columns:
            s6 += "\n{\nname: '" + str(name) + "',\ndata:"
            tmp = new_two_col(profits, name)
            s6 += tmp[['rets100', 'hist']
                      ].to_json(orient='values', double_precision=4)
            s6 += "\n},"
        s6 += "\n]\n"
    return s6


# ==============================================================================
# 箱线图
# ==============================================================================
def get_quantile(profits, name):
    IQR = (profits[name].quantile(q=0.75) -
           profits[name].quantile(q=0.25)) * 1.5
    up1 = profits[name][profits[name] <= (
        profits[name].quantile(q=0.75) + IQR)]
    down1 = profits[name][profits[name] >= (
        profits[name].quantile(q=0.25) - IQR)]
    return np.intersect1d(
        up1, down1).min(), profits[name].quantile(
        q=0.25), profits[name].median(), profits[name].quantile(
            q=0.75), np.intersect1d(
                up1, down1).max()


def Box_Retun(profits):
    if profits.columns.size == 1:
        s7 = "\nvar d_Box_DailyReturn = [["
        up, quantile3, median_num, quantile1, down = get_quantile(
            profits, profits.columns[0])

        s7 += format(down * 100,'.4f') + "," + format(quantile1 * 100,'.4f') + "," + format(median_num * 100,
                    '.4f') + "," + format(quantile3 * 100,'.4f') + "," + format(up * 100,'.4f') + "]];"
        s7 += "\n"
    else:
        s7 = "\nvar d_Box_DailyReturn_series = [\n{\n"
        s7 += "categories: " + str(profits.columns.tolist()) + ",\n"
        s7 += "data:["
        for name in profits.columns:
            down, quantile1, median_num, quantile3, up = get_quantile(
                profits, name)

            s7 += "[" + format(down * 100,'.4f') + "," + format(quantile1 * 100,'.4f') + "," + format(median_num * 100,
                    '.4f') + "," + format(quantile3 * 100,'.4f') + "," + format(up * 100,'.4f') + "],"

        s7 += "]\n},\n]\n"
    return s7


# ==============================================================================
# 回撤图
# ==============================================================================
def areaChart_DrawDown(navs):
    if navs.columns.size == 1:
        drawdown = (navs - np.maximum.accumulate(navs)) / \
            np.maximum.accumulate(navs)
        s8 = "\nvar d_areaChart_DrawDown="
        s8 += drawdown.reset_index().to_json(orient='values', double_precision=4)
        s8 += "\n"
    else:
        s8 = "\nvar d_areaChart_DrawDown=["
        for name in navs.columns:
            navs1 = navs[name].dropna()
            drawdown = (navs1 - np.maximum.accumulate(navs1)) / \
                np.maximum.accumulate(navs1)
            s8 += "\n{\nname: '" + str(name) + "',\ndata:"
            s8 += (drawdown * 100).reset_index().to_json(orient='values',
                                                         double_precision=4)
            s8 += "\n},"
        s8 += "\n]\n"
    return s8


# ==============================================================================
# 滚动年化收益
# ==============================================================================
def data_gundongyuedushouyi(navs_and_date, p):
    if navs_and_date.columns.size == 1:
        rollingmonth3m = rolling_month_return_3m(navs_and_date, p[0])
        rollingmonth1m = rolling_month_return_1m(navs_and_date, p[0])
        s9 = "\nvar d_gundongnianhua=["
        s9 += "\n{\nname: '" + "滚动年化收益率-1月" + "',\ndata:"
        s9 += (rollingmonth1m *
               100).reset_index().to_json(orient='values', double_precision=2)
        s9 += "\n},"
        s9 += "\n{\nname: '" + "滚动年化收益率-3月" + "',\ndata:"
        s9 += (rollingmonth3m *
               100).reset_index().to_json(orient='values', double_precision=2)
        s9 += "\n},"

        s9 += "\n]\n"
    else:
        s9 = "\nvar d_gundongyuedushouyi_series=["
        for i, name in enumerate(navs_and_date.columns):
            navs1 = navs_and_date[name].dropna()
            rollingmonth = rolling_month_return_3m(navs1, p[i])
            s9 += "\n{\nname: '" + str(name) + "',\ndata:"
            s9 += rollingmonth.reset_index().to_json(orient='values', double_precision=4)
            s9 += "\n},"
        s9 += "\n]\n"
    return s9


def rolling_month_return_3m(nav, p):
    def rolling_month_return_apply(x):
        return (x[-1] / x[0] - 1) * 365 / 91

    period = {'日': 61, '周': 13, '月': 4, '残缺': 4}
    result = nav.rolling(window=period[p]).apply(rolling_month_return_apply)
    return result.fillna(0)


def rolling_month_return_1m(nav, p):
    def rolling_month_return_apply(x):
        return (x[-1] / x[0] - 1) * 365 / 30

    period = {'日': 21, '周': 5, '月': 2, '残缺': 2}
    result = nav.rolling(window=period[p]).apply(rolling_month_return_apply)
    return result.fillna(0)



# ==============================================================================
# 滚动年化波动
# ==============================================================================
def rolling_fluctuation_year_3m(profits, p):  # for pandas
    period = {'日': [61, 242], '周': [13, 48],
              '月': [4, 12], '残缺': [4, 12]}
    result = profits.rolling(
        window=period[p][0], center=False).std() * np.sqrt(period[p][1])
    return result.fillna(0)


def rolling_fluctuation_year_1m(profits, p):  # for pandas
    period = {'日': [21, 242], '周': [5, 48],
              '月': [2, 12], '残缺': [2, 12]}
    result = profits.rolling(
        window=period[p][0], center=False).std() * np.sqrt(period[p][1])
    return result.fillna(0)


def data_gundongnianhua_vol(profits, p):
    if profits.columns.size == 1:
        rolling_fluctuation3m = rolling_fluctuation_year_3m(
            profits, p[0]) * 100
        rolling_fluctuation1m = rolling_fluctuation_year_1m(
            profits, p[0]) * 100
        s10 = "\nvar d_gundong_vol=["
        s10 += "\n{\nname: '" + "滚动年化波动率-1月" + "',\ndata:"
        s10 += rolling_fluctuation1m.reset_index().to_json(orient='values',
                                                           double_precision=2)
        s10 += "\n},"
        s10 += "\n{\nname: '" + "滚动年化波动率-3月" + "',\ndata:"
        s10 += rolling_fluctuation3m.reset_index().to_json(orient='values',
                                                           double_precision=2)
        s10 += "\n},"

        s10 += "\n]\n"
    else:
        s10 = "\nvar d_gundong_vol=["
        for i, name in enumerate(profits.columns):
            profits1 = profits[name].dropna()
            rolling_fluctuation = rolling_fluctuation_year_3m(profits1, p[i])
            s10 += "\n{\nname: '" + str(name) + "',\ndata:"
            s10 += (rolling_fluctuation *
                    100).reset_index().to_json(orient='values', double_precision=4)
            s10 += "\n},"
        s10 += "\n]\n"
    return s10



def Corr(rets):
    """
    相关性矩阵计算并转换为列表格式
    """
    l = []
    cor = rets.corr(method='spearman').fillna(0)
    for i in range(cor.shape[0]):
        for j in range(cor.shape[1]):
            l.append([i, j, round(cor.ix[i, j], 2)])
    s15 = "\nvar d_heatmapdata="
    s15 += str(l)

    s15 += "\nvar d_namesx="
    s15 += str(cor.columns.tolist())

    s15 += "\nvar d_namesy="
    s15 += str(cor.index.tolist())

    return s15



def Heatmap_multi(rets, strategy):
    """
    相关性矩阵产品和策略对比版
    """
    # 从mysql中读取数据
    if '型' not in strategies_dict[strategy[0]]:
        clzs = get_data_from_mysql('select index_date, \
                                    all_market as 全市场策略, \
                                    macro as 宏观对冲策略, \
                                    relative as 相对价值策略, \
                                    cta as 管理期货策略, \
                                    stocks as 股票多头策略, \
                                    bonds as 固定收益策略  from t_strategy_index', 'index_date')
    else:
        clzs = get_data_from_mysql('select index_date, \
                                    commingled as 混合型, \
                                    bonds as 债券型, \
                                    money as 货币型, \
                                    stocks as 股票型 \
                                    from t_public_strategy_index', 'index_date')
    clzs.index = pd.to_datetime(clzs.index)
    # 从mysql中读取数据
    table = get_data_from_mysql('select index_date, \
                                hs300 as 沪深300指数, \
                                csi_total_debt as 中证全债指数, \
                                south_china_commodity as 南华商品指数 from t_wind_index_market', 'index_date')
    table.index = pd.to_datetime(table.index)
    cl = clzs[list(set([strategies_dict[i] for i in list(set(strategy))]))]
    sc = table[list(set([market_dict[i] for i in list(set(strategy))]))]
    data = pd.concat([rets, cl, sc], axis=1, join='outer')
    l = []
    for i in range(data.columns.size):
        for j in range(data.columns.size):
            l.append([i, j, round(data.ix[:, [i, j]].dropna().corr(
                method='spearman').fillna(0).values[0, 1], 3)])
    s = "\nvar d_heatmapdata="
    s += str(l)

    s += "\nvar d_namesx="
    s += str(data.columns.tolist())

    s += "\nvar d_namesy="
    s += str(data.columns.tolist())
    return s

#张焕芳编写
def Heatmap_multi3(rets, strategy):
    """
    相关性矩阵产品和策略以及市场对比版
    """
    # 从mysql中读取数据
    clzs_simu = get_data_from_mysql('select index_date, \
                                    all_market as 全市场策略, \
                                    macro as 宏观对冲策略, \
                                    relative as 相对价值策略, \
                                    cta as 管理期货策略, \
                                    stocks as 股票多头策略, \
                                    bonds as 固定收益策略  from t_strategy_index', 'index_date')
   
    clzs_public = get_data_from_mysql('select index_date, \
                                    commingled as 混合型, \
                                    bonds as 债券型, \
                                    money as 货币型, \
                                    stocks as 股票型 \
                                    from t_public_strategy_index', 'index_date')
    clzs_simu.index = pd.to_datetime(clzs_simu.index)
    clzs_public.index = pd.to_datetime(clzs_public.index)
    # 从mysql中读取数据
    table = get_data_from_mysql('select index_date, \
                                hs300 as 沪深300指数, \
                                csi_total_debt as 中证全债指数, \
                                south_china_commodity as 南华商品指数 from t_wind_index_market', 'index_date')
    table.index = pd.to_datetime(table.index)
    
    clzs=pd.concat([clzs_simu,clzs_public], axis=1, join='outer')
 
    cl = clzs[list(set([strategies_dict[i] for i in list(set(strategy))]))]
    cl_rets=calculate_profits(cl)
    sc = table[list(set([market_dict[i] for i in list(set(strategy))]))]
    sc_rets=calculate_profits(sc)
    data = pd.concat([ sc_rets, cl_rets,rets], axis=1, join='outer')
     
    corr = pd.DataFrame()
    for i in range(data.columns.size):
        m=[]
        for j in range(data.columns.size):
            m.append([round(data.ix[:, [i, j]].dropna().corr(
                method='spearman').fillna(0).values[0, 1], 3)])
        corr=pd.concat([corr,pd.DataFrame(m).T])    
    corr.columns=data.columns  
    corr.index=data.columns         
    return corr

def Heatmap_multi_4(rets,strategy):
    """
    相关性矩阵多产品版,根据需求当产品数量较多时，只求产品间以及市场指数的相关系数 不须考虑策略指数 
    """
    table = get_data_from_mysql('select index_date, \
                                hs300 as 沪深300指数, \
                                csi_total_debt as 中证全债指数, \
                                south_china_commodity as 南华商品指数 from t_wind_index_market', 'index_date')
    table.index = pd.to_datetime(table.index)
    sc = table[list(set([market_dict[i] for i in list(set(strategy))]))]
    sc_rets=calculate_profits(sc)
    data = pd.concat([sc_rets,rets], axis=1, join='outer')
    corr = pd.DataFrame()
    for i in range(data.columns.size):
        m=[]
        for j in range(data.columns.size):
            m.append([round(data.ix[:, [i, j]].dropna().corr(
                method='spearman').fillna(0).values[0, 1], 3)])
        corr=pd.concat([corr,pd.DataFrame(m).T])    
    corr.columns=data.columns  
    corr.index=data.columns         
    return  corr

def Heatmap_multi_5(rets):
    """
    相关性矩阵多产品版,根据需求当产品数量较多时，只求产品间的相关系数 不须考虑市场和策略指数 
    """
    corr = pd.DataFrame()
    for i in range(rets.columns.size):
        m=[]
        for j in range(rets.columns.size):
            m.append([round(rets.ix[:, [i, j]].dropna().corr(
                method='spearman').fillna(0).values[0, 1], 3)])
        corr=pd.concat([corr,pd.DataFrame(m).T])    
    corr.columns=rets.columns  
    corr.index=rets.columns         
    return  corr


'''下面这个函数是朱晨曦加的：根据需求当产品数量较多时，只求产品间的相关系数 不须考虑市场  若发现有问题请修改 '''


def Heatmap_multi_2(rets):
    """
    相关性矩阵多产品版
    """
    l = []
    for i in range(rets.columns.size):
        for j in range(rets.columns.size):
            l.append([i, j, round(rets.ix[:, [i, j]].dropna().corr(
                method='spearman').fillna(0).values[0, 1], 3)])
    s = "\nvar d_heatmapdata="
    s += str(l)

    s += "\nvar d_namesx="
    s += str(rets.columns.tolist())

    s += "\nvar d_namesy="
    s += str(rets.columns.tolist())
    return s



def dealTd(value, No, flag=0):
    """
    拼字符串，输出html表格，横版
    """
    s = ''
    for i in range(No):
        if isinstance(value[i], str):
            s += "<td>{}</td>".format(value[i])
        elif flag == 0:
            s += "<td>{:.2f}</td>".format(value[i])
        else:
            s += "<td>{:.2%}</td>".format(value[i])
    return s


def dealTd_h(value, No, flag=0):
    """
    拼字符串，输出html表格，竖版
    """
    s = "<tr>"
    for i in range(No):
        if isinstance(value[i], str):
            s += "<td>{}</td>".format(value[i])
        elif flag == 0:
            s += "<td>{:.2f}</td>".format(value[i])
        else:
            s += "<td>{:.2%}</td>".format(value[i])
    s += "</tr>"
    return s


def Radar(annRet,annVol,maxdown,mean_drawdown,calmar,stutzer,var,updown,win,strategy):
    """
    雷达图
    """
    if strategies_dict[strategy] in [
        '宏观对冲策略',
        '债券策略',
        '相对价值策略',
        '管理期货策略',
            '股票多头策略']:
        strategy = strategies_dict[strategy]
    else:
        strategy = '宏观对冲策略'
    # 从mysql中读取数据
    the_sql = "select bound,return_annual as 年化收益,volatility_annual as 年化波动,\
    drawback_max as 最大回撤, drawback_avg as 平均回撤,calmar as 卡玛,stutzer as 斯图泽,\
    var as VAR, up_ratio as 上行比例,win_ratio as 胜赢率 from t_strategy_bound where strategy_type = '" + strategy + "'"
    bound = get_data_from_mysql(the_sql, 'bound')

    def bound_limit(a):
        if a != a:
            return 0
        return int(np.where(a > 100, 100, np.where(a < 0, 0, a)))

    radar_annRet = bound_limit(
        (annRet[0] - bound.ix['down', 0]) / (bound.ix['up', 0] - bound.ix['down', 0]) * 100)
    radar_vol = bound_limit((annVol[0] -
                             bound.ix['down', 1]) /
                            (bound.ix['up', 1] -
                             bound.ix['down', 1]) *
                            100)
    radar_mdd = bound_limit((maxdown[0] -
                             bound.ix['down', 2]) /
                            (bound.ix['up', 2] -
                             bound.ix['down', 2]) *
                            100)
    radar_mean_drawdown = bound_limit(
        (mean_drawdown[0] - bound.ix['down', 3]) / (bound.ix['up', 3] - bound.ix['down', 3]) * 100)
    radar_calmar = bound_limit(
        (calmar[0] - bound.ix['down', 4]) / (bound.ix['up', 4] - bound.ix['down', 4]) * 100)
    radar_stutzer = bound_limit(
        (stutzer[0] - bound.ix['down', 5]) / (bound.ix['up', 5] - bound.ix['down', 5]) * 100)
    radar_var = bound_limit(
        (var[0] - bound.ix['down', 6]) / (bound.ix['up', 6] - bound.ix['down', 6]) * 100)
    radar_updown = bound_limit(
        (updown[0] - bound.ix['down', 7]) / (bound.ix['up', 7] - bound.ix['down', 7]) * 100)
    radar_win = bound_limit(
        (win[0] - bound.ix['down', 8]) / (bound.ix['up', 8] - bound.ix['down', 8]) * 100)
    group1_ret = radar_annRet
    group2_risk = (radar_mdd + radar_var) / 2
    group3_risk_adj = (radar_calmar + radar_stutzer) / 2
    group4_stability = (radar_mean_drawdown + radar_vol) / 2
    group5_potential = (radar_updown + radar_win) / 2
    l = list([group1_ret, group2_risk, group3_risk_adj,
              group4_stability, group5_potential])
    return l

def Radar_CTA(annRet,annVol,maxdown,mean_drawdown,calmar,stutzer,var,updown,strategy):
    """
    雷达图CTA归因版，未加入胜率
    """
    if strategies_dict[strategy] in [
        '宏观对冲策略',
        '债券策略',
        '相对价值策略',
        '管理期货策略',
            '股票多头策略']:
        strategy = strategies_dict[strategy]
    else:
        strategy = '宏观对冲策略'
    # 从mysql中读取数据
    the_sql = "select bound, return_annual as 年化收益,volatility_annual as 年化波动, drawback_max as 最大回撤,drawback_avg as 平均回撤,calmar as 卡玛,stutzer as 斯图泽,var as VAR,up_ratio as 上行比例 from t_strategy_bound where strategy_type = '" + strategy + "'"
    bound = get_data_from_mysql(the_sql, 'bound')

    def bound_limit(a):
        if a != a:
            return 0
        return int(np.where(a > 100, 100, np.where(a < 0, 0, a)))

    radar_annRet = bound_limit(
        (annRet[0] - bound.ix['down', 0]) / (bound.ix['up', 0] - bound.ix['down', 0]) * 100)
    radar_vol = bound_limit((annVol[0] -
                             bound.ix['down', 1]) /
                            (bound.ix['up', 1] -
                             bound.ix['down', 1]) *
                            100)
    radar_mdd = bound_limit((maxdown[0] -
                             bound.ix['down', 2]) /
                            (bound.ix['up', 2] -
                             bound.ix['down', 2]) *
                            100)
    radar_mean_drawdown = bound_limit(
        (mean_drawdown[0] - bound.ix['down', 3]) / (bound.ix['up', 3] - bound.ix['down', 3]) * 100)
    radar_calmar = bound_limit(
        (calmar[0] - bound.ix['down', 4]) / (bound.ix['up', 4] - bound.ix['down', 4]) * 100)
    radar_stutzer = bound_limit(
        (stutzer[0] - bound.ix['down', 5]) / (bound.ix['up', 5] - bound.ix['down', 5]) * 100)
    radar_var = bound_limit(
        (var[0] - bound.ix['down', 6]) / (bound.ix['up', 6] - bound.ix['down', 6]) * 100)
    radar_updown = bound_limit(
        (updown[0] - bound.ix['down', 7]) / (bound.ix['up', 7] - bound.ix['down', 7]) * 100)
    group1_ret = radar_annRet
    group2_risk = (radar_mdd + radar_var) / 2
    group3_risk_adj = (radar_calmar + radar_stutzer) / 2
    group4_stability = (radar_mean_drawdown + radar_vol) / 2
    group5_potential = radar_updown
    l = list([group1_ret, group2_risk, group3_risk_adj,
              group4_stability, group5_potential])
    return l
# ==============================================================================
# 下行风险
# ==============================================================================
def downsideRisk2(profit, return_noRisk_new):
    """
    下行风险
    """
    neg = profit[profit < return_noRisk_new]
    if len(neg) >= 2:
        return np.sqrt(((neg - return_noRisk_new) ** 2).sum() / (len(neg) - 1))
    else:
        return -1


# 打分
def det(
       annret,
        alpha_ret,
        sortino,
        omega,
        Calmar,
        M2,
        mean_drawdown,
        annvol,
        downrisk,
        Cvar95,
        CL_alpha,
        HM_alpha,
        TM_alpha,
        CL_beta_ratio,
        HM_beta2,
        TM_beta2,
        beta_ratio,
        saomiao,
        industry_ret,zhibiao_stock):
        
    #各项指标单项评
        score=pd.Series()
        score['年化收益']=((pd.Series(zhibiao_stock['年化收益']).astype(float)-annret)<0).sum()/len(zhibiao_stock)*100
        score['超额收益']=((pd.Series(zhibiao_stock['超额收益']).astype(float)-alpha_ret)<0).sum()/len(zhibiao_stock)*100
        score['索提诺比率']=((pd.Series(zhibiao_stock['索提诺比率']).astype(float)-sortino)<0).sum()/len(zhibiao_stock)*100
        score['Omega比率']=((pd.Series(zhibiao_stock['Omega比率']).astype(float)-omega)<0).sum()/len(zhibiao_stock)*100
        score['卡玛比']=((pd.Series(zhibiao_stock['卡玛比']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_stock)*100
        score['M平方']=((pd.Series(zhibiao_stock['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_stock)*100
        score['年化波动率']=((pd.Series(zhibiao_stock['年化波动率']).astype(float)-annvol)>0).sum()/len(zhibiao_stock)*100
        score['下行风险']=((pd.Series(zhibiao_stock['下行风险']).astype(float)-downrisk)>0).sum()/len(zhibiao_stock)*100
        score['平均回撤']=((pd.Series(zhibiao_stock['平均回撤']).astype(float)-mean_drawdown)>0).sum()/len(zhibiao_stock)*100
        score['Cvar95']=((pd.Series(zhibiao_stock['Cvar95']).astype(float)-Cvar95)>0).sum()/len(zhibiao_stock)*100
        score['行业配置能力']= 60 + industry_ret*400
        score['CL择股能力']=((pd.Series(zhibiao_stock['CL择股能力']).astype(float)- CL_alpha)<0).sum()/len(zhibiao_stock)*100
        score['HM择股能力']=((pd.Series(zhibiao_stock['HM择股能力']).astype(float)- HM_alpha)<0).sum()/len(zhibiao_stock)*100
        score['TM择股能力']=((pd.Series(zhibiao_stock['TM择股能力']).astype(float)- TM_alpha)<0).sum()/len(zhibiao_stock)*100
        score['CL择时能力']=((pd.Series(zhibiao_stock['CL择时能力']).astype(float)- CL_beta_ratio)<0).sum()/len(zhibiao_stock)*100
        score['HM择时能力']=((pd.Series(zhibiao_stock['HM择时能力']).astype(float)- HM_beta2)<0).sum()/len(zhibiao_stock)*100
        score['TM择时能力']=((pd.Series(zhibiao_stock['TM择时能力']).astype(float)- TM_beta2)<0).sum()/len(zhibiao_stock)*100
        score['贝塔择时能力']=((pd.Series(zhibiao_stock['beta择时能力']).astype(float)- beta_ratio)<0).sum()/len(zhibiao_stock)*100
        score['业绩一致性']=((pd.Series(zhibiao_stock['扫描统计量']).astype(float)- saomiao)<0).sum()/(zhibiao_stock['扫描统计量'].dropna().size)*100
        score_radar=pd.Series()
        score_radar['收益能力']= score['年化收益']*0.5 + score['超额收益']*0.5
        score_radar['择时能力']= (score['CL择时能力']+score['HM择时能力']+score['TM择时能力']+ score['贝塔择时能力'])/4
        score_radar['择股能力']=(score['CL择股能力']+ score['HM择股能力']+ score['TM择股能力'])/3
        score_radar['行业配置能力']= score['行业配置能力']
        score_radar['风控能力']= (score['年化波动率']+score['下行风险']+score['平均回撤']+ score['Cvar95'])/4
        score_radar['风险调整绩效']= (score['索提诺比率']+ score['Omega比率']+score['卡玛比']+ score['M平方'])/4
        score_radar['业绩持续性']=  score['业绩一致性']
        score_radar['综合能力']= score_radar['收益能力']*0.2+ score_radar['择时能力']*0.1+ score_radar['择股能力']*0.1+  score_radar['行业配置能力']*0.1+\
        score_radar['风控能力']*0.2+ score_radar['风险调整绩效']*0.2 +score_radar['业绩持续性']*0.1#+#score_radar['持续性']*0.1
    #星级评定
        star_level=[]
        for i in range(score_radar.size):
            if score_radar[i]>=90:  
                scar_level=5
            elif score_radar[i]<90 and score_radar[i]>=80:
                 scar_level=4.5
            elif score_radar[i]<80 and score_radar[i]>=70:
                 scar_level=4
            elif score_radar[i]<70 and score_radar[i]>=60:
                scar_level=3.5
            elif score_radar[i]>=50 and score_radar[i]<60:
                scar_level=3
            elif score_radar[i]>=40 and score_radar[i]<50:  
                scar_level=2.5
            else:
                scar_level=2          
            star_level.append(scar_level)           
        star_table=pd.Series(star_level,index=['收益能力','择时能力','择股能力','行业配置能力','风控能力','风险调整绩效','业绩持续性','综合能力1'])  
        return score,score_radar,star_table



def alpha_calc_data(navs,p,con):
        indicatrix_data = {}
        period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
        hs300 =fetch_index(con)
#    p_judge= {'日': 0, '周': 1, '月': 1, '残缺': 0}
#    hs300 = pd.DataFrame(hs300.iloc[:, p_judge[p[0]]])
        hs300 = pd.DataFrame(hs300.iloc[:, 0])
        hs300.rename(columns={'沪深300-日':'沪深300'},inplace=True)
        indicatrix_data['market300'] = hs300
        profits=calculate_profits(navs)
#        navs=combin
        indicatrix_data['the_navs'], indicatrix_data['300index_navs'] =interpolate(
            pd.DataFrame(navs), pd.DataFrame(indicatrix_data['market300']), p)
        indicatrix_data['300index_pct'] =calculate_profits(
            indicatrix_data['300index_navs'])
        profits=calculate_profits(navs)
        indicatrix_data['the_pct']=calculate_profits(indicatrix_data['the_navs'])
        indicatrix_data['data_annRets'] =annRets(navs)
        indicatrix_data['data_annVol'] =annVol(profits, p)
        indicatrix_data['data_maxdrawdown'] = maxdrawdown(navs)
        indicatrix_data['data_mean_drawdown'] =meandrawdown(navs)
        indicatrix_data['data_Sharp'] =SharpRatio(profits, p)
        indicatrix_data['data_Calmar'] =Calmar(indicatrix_data['data_annRets'], indicatrix_data['data_maxdrawdown'])
        indicatrix_data['data_inforatio']=InfoRation2(indicatrix_data['the_pct'],indicatrix_data['300index_pct'],['日'])
        
        indicatrix_data['data_Sortino'] =Sortino(profits, p)
        indicatrix_data['data_Stutzer'] =Stutzer2(profits, p)
        indicatrix_data['data_var95'] =VARRatio(0.95, profits,p)
#        indicatrix_data['data_var99'] = functions.VARRatio(0.99, profits)
        indicatrix_data['data_cvar95'] =CVar(profits, 0.95,p)
        indicatrix_data['data_alpha_famma']=fama(navs)
        
        #下行风险
        indicatrix_data['downrisk'] =downsideRisk2(indicatrix_data['the_pct'], return_noRisk/period[p[0]])*np.sqrt(period[p[0]])
        
#        indicatrix_data['data_cvar99'] = functions.CVar(profits, 0.99)
        indicatrix_data['data_saomiao']=saomiao2(profits,p)
        indicatrix_data['data_WinRate'] =WinRate(profits)
        _,indicatrix_data['data_signaldrawdown']=signal_drawdown(navs)
        indicatrix_data['omega'] = Omega(profits,p)
        indicatrix_data['M2'] =M2(indicatrix_data['the_navs'],indicatrix_data['300index_pct'], p)
        
        indicatrix_data['data_alpha'], indicatrix_data['data_beta'], indicatrix_data['data_R2'] =alpha_and_beta(
            indicatrix_data['the_pct'], indicatrix_data['300index_pct'],p)
       
        [HM_alpha,HM_beta1,HM_beta2,HM_p_value]=HM2(pd.DataFrame( indicatrix_data['the_pct']), pd.DataFrame(indicatrix_data['300index_pct']),p)
        CL= CL2(pd.DataFrame( indicatrix_data['the_pct']), pd.DataFrame(indicatrix_data['300index_pct']), p)
        [TM_alpha,TM_beta1,TM_beta2,TM_t_test]= TM2(pd.DataFrame( indicatrix_data['the_pct']), pd.DataFrame(indicatrix_data['300index_pct']), p)

        indicatrix_data['data_HM_alpha']=HM_alpha[0]*period[p[0]]
        indicatrix_data['data_TM_alpha']=TM_alpha[0]*period[p[0]]
        indicatrix_data['data_CL_alpha']= CL[0][0]*period[p[0]]
        indicatrix_data['CL_alpha_p_value']= CL[0][3][1][0]
        indicatrix_data['CL_beta_p_value']=CL[0][3][1][1]
        indicatrix_data['HM_alpha_p_value']=HM_p_value[1][0]
        indicatrix_data['HM_beta_p_value']=HM_p_value[1][1]
        indicatrix_data['TM_alpha_p_valu']=TM_t_test[1][0]
        indicatrix_data['TM_beta_p_value']=TM_t_test[1][1]
        indicatrix_data['CL_beta_ratio']= CL[0][2]- CL[0][1]
        indicatrix_data['HM_beta2'] =HM_beta2
        indicatrix_data['TM_beta2'] =TM_beta2
        
        beta_diff, params_niu, params_xiong = differ_market(indicatrix_data['the_pct'], indicatrix_data['300index_pct'])
        indicatrix_data['niu_beta']=params_niu[0]
        indicatrix_data['xiong_beta']=params_xiong[0]
        #贝塔择时比率
        indicatrix_data['beta_zeshi']= indicatrix_data['niu_beta']-indicatrix_data['xiong_beta']
    
        return  indicatrix_data

def ranking(strategy_type,return_annual,drawback_max,volatility_annual,sharpe,Calmar):
    """
    产品打分
    """
    score = 0
    if strategy_type in ('宏观策略' , '宏观对冲' , '全球宏观','复合策略'):
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
    elif strategy_type in ('相对价值' , '市场中性' , '量化对冲' , '股票对冲'):
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.1, 0, -0.01, -0.03, 0.07, 0.13)
    elif strategy_type in ('管理期货' , 'CTA策略' , 'CTA'):
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.15, 0.05, -0.03, -0.07, 0.1, 0.16)
    elif strategy_type in ('债券策略' , '债券型','固定收益'):
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.08, 0.055, -0.005, -0.015, 0.03, 0.07)
    elif strategy_type in ('股票多头' , '股票型' , '股票策略' , '股票量化'):
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
    else:
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
    return score
#以下为新版打分，编写
#对股票策略打分
def det_stock(*args):
    if len(args)==9:
        annret,alpha_ret,sortino,Calmar,M2,signal_drawdown,winrate,Cvar95,zhibiao_strategy=args
        score=pd.Series()
        score['年化收益']=((pd.Series(zhibiao_strategy['anyet']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
        score['超额收益']=((pd.Series(zhibiao_strategy['阿尔法']).astype(float)-alpha_ret)<0).sum()/len(zhibiao_strategy)*100
        score['索提诺比率']=((pd.Series(zhibiao_strategy['索提诺']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
        score['卡玛比']=((pd.Series(zhibiao_strategy['卡玛比']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
        score['M平方']=((pd.Series(zhibiao_strategy['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
        score['单一最大回撤']=((pd.Series(zhibiao_strategy['单一最大回撤']).astype(float)-signal_drawdown)>0).sum()/len(zhibiao_strategy)*100
        score['Cvar95']=((pd.Series(zhibiao_strategy['CVaR(95%)']).astype(float)-Cvar95)>0).sum()/len(zhibiao_strategy)*100
       
        score['胜率']=((pd.Series(zhibiao_strategy['胜率']).astype(float)-winrate)<0).sum()/len(zhibiao_strategy)*100
        score_radar=pd.Series()
        score_radar['收益能力']= score['年化收益']
        score_radar['净值管理能力']=  score['单一最大回撤']
        score_radar['业绩持续性']=score['胜率']
    
        score_radar['风险调整绩效']= (score['索提诺比率']+score['卡玛比']+ score['M平方'])/3
        score_radar['风控能力']=  score['Cvar95']
        score_radar['选股能力']=   score['超额收益']
    
        score_radar['综合能力']= score_radar['收益能力']*0.3+  score_radar['净值管理能力']*0.06+\
        score_radar['风控能力']*0.1+ score_radar['风险调整绩效']*0.38 +score_radar['业绩持续性']*0.1+ score_radar['选股能力']*0.06#+#score_radar['持续性']*0.1
    elif len(args)==19:
        annret,alpha_ret, sortino,omega,Calmar,M2,mean_drawdown,annvol,downrisk,Cvar95,CL_alpha,HM_alpha,TM_alpha, \
        CL_beta_ratio,HM_beta2,TM_beta2,beta_ratio,saomiao,zhibiao_strategy=args
        score=pd.Series()
        score['年化收益']=((pd.Series(zhibiao_strategy['return_annual']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
        score['超额收益']=((pd.Series(zhibiao_strategy['alpha']).astype(float)-alpha_ret)<0).sum()/len(zhibiao_strategy)*100
        score['索提诺比率']=((pd.Series(zhibiao_strategy['sortino']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
        score['Omega比率']=((pd.Series(zhibiao_strategy['omega']).astype(float)-omega)<0).sum()/len(zhibiao_strategy)*100
        score['卡玛比']=((pd.Series(zhibiao_strategy['calmar']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
        score['M平方']=((pd.Series(zhibiao_strategy['M2']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
        score['年化波动率']=((pd.Series(zhibiao_strategy['volatility_annual']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
        score['下行风险']=((pd.Series(zhibiao_strategy['downrisk']).astype(float)-downrisk)>0).sum()/len(zhibiao_strategy)*100
        score['平均回撤']=((pd.Series(zhibiao_strategy['data_mean_drawdown']).astype(float)-mean_drawdown)>0).sum()/len(zhibiao_strategy)*100
        score['Cvar95']=((pd.Series(zhibiao_strategy['data_CVaR95']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
        score['CL择股能力']=((pd.Series(zhibiao_strategy['CL_alpha']).astype(float)- CL_alpha)<0).sum()/len(zhibiao_strategy)*100
        score['HM择股能力']=((pd.Series(zhibiao_strategy['HM_alpha']).astype(float)- HM_alpha)<0).sum()/len(zhibiao_strategy)*100
        score['TM择股能力']=((pd.Series(zhibiao_strategy['TM_alpha']).astype(float)- TM_alpha)<0).sum()/len(zhibiao_strategy)*100
        score['CL择时能力']=((pd.Series(zhibiao_strategy['CL_beta_ratio']).astype(float)- CL_beta_ratio)<0).sum()/len(zhibiao_strategy)*100
        score['HM择时能力']=((pd.Series(zhibiao_strategy['HM_beta2']).astype(float)- HM_beta2)<0).sum()/len(zhibiao_strategy)*100
        score['TM择时能力']=((pd.Series(zhibiao_strategy['TM_beta2']).astype(float)- TM_beta2)<0).sum()/len(zhibiao_strategy)*100
        score['贝塔择时能力']=((pd.Series(zhibiao_strategy['beita_zeshi_all']).astype(float)- beta_ratio)<0).sum()/len(zhibiao_strategy)*100
        score['业绩一致性']=((pd.Series(zhibiao_strategy['saomiao']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['saomiao'].dropna().size)*100  
        score_radar=pd.Series()
        score_radar['收益能力']= score['年化收益']*0.5 + score['超额收益']*0.5
        score_radar['择时能力']= (score['CL择时能力']+score['HM择时能力']+score['TM择时能力']+ score['贝塔择时能力'])/4
        score_radar['择股能力']=(score['CL择股能力']+ score['HM择股能力']+ score['TM择股能力'])/3
        score_radar['风控能力']= (score['年化波动率']+score['下行风险']+score['平均回撤']+ score['Cvar95'])/4
        score_radar['风险调整绩效']= (score['索提诺比率']+ score['Omega比率']+score['卡玛比']+ score['M平方'])/4
        score_radar['业绩持续性']=  score['业绩一致性']
    #    score_radar['行情判断能力']= score['胜率'] 
    #    score_scar=pd.Series()
    #    score_star=pd.Series()
        score_radar['综合能力']= score_radar['收益能力']*0.25+ score_radar['择时能力']*0.1+ score_radar['择股能力']*0.1+\
        score_radar['风控能力']*0.2+ score_radar['风险调整绩效']*0.2 +score_radar['业绩持续性']*0.15#+#score_radar['持续性']*0.1
        #星级评定
        star_level=[]
        for i in range(score_radar.size):
            if score_radar[i]>=90:  
                    scar_level=5
            elif score_radar[i]<90 and score_radar[i]>=80:
                     scar_level=4.5
            elif score_radar[i]<80 and score_radar[i]>=70:
                     scar_level=4
            elif score_radar[i]<70 and score_radar[i]>=60:
                    scar_level=3.5
            elif score_radar[i]>=50 and score_radar[i]<60:
                    scar_level=3
            elif score_radar[i]>=40 and score_radar[i]<50:  
                    scar_level=2.5
            else:
                    scar_level=2          
            star_level.append(scar_level)           
        star_table=pd.Series(star_level,index=['收益能力','择时能力','择股能力','风控能力','风险调整绩效','业绩持续性','综合能力1']) 
    #    star_table['相对收益能力']='--'
        return score,score_radar, star_table        

    
def fof_judge(annret,sortino, omega,Calmar,M2,mean_drawdown,annvol, downrisk,Cvar95,saomiao,zhibiao_strategy):
    
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['anyet']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['索提诺比率']=((pd.Series(zhibiao_strategy['索提诺']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
    score['Omega比率']=((pd.Series(zhibiao_strategy['omega']).astype(float)-omega)<0).sum()/len(zhibiao_strategy)*100
    score['卡玛比']=((pd.Series(zhibiao_strategy['卡玛比']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
    score['年化波动率']=((pd.Series(zhibiao_strategy['年化波动率']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
    score['下行风险']=((pd.Series(zhibiao_strategy['下行风险']).astype(float)-downrisk)>0).sum()/len(zhibiao_strategy)*100
    score['平均回撤']=((pd.Series(zhibiao_strategy['平均回撤']).astype(float)-mean_drawdown)>0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['CVaR(95%)']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
# 

    #胜率评分
#    if  win_ratio>=0.7:
#         score['胜率']=100     
#    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
#         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
#    elif win_ratio<=0.31:
#         score['胜率']=0
    score['业绩一致性']=((pd.Series(zhibiao_strategy['扫描统计量']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['扫描统计量'].dropna().size)*100
    score_radar=pd.Series()
    score_radar['收益能力']= score['年化收益']
    score_radar['风控能力']= (score['年化波动率']+score['下行风险']+score['平均回撤']+ score['Cvar95'])/4
    score_radar['风险调整绩效']= (score['索提诺比率']+ score['Omega比率']+score['卡玛比']+ score['M平方'])/4
    score_radar['业绩持续性']=  score['业绩一致性']
#    score_radar['择时能力']='--' 
#    score_radar['择股能力']='--'
#    score_radar['行情判断能力']= score['胜率']
#    score_scar=pd.Series()
    score_radar['综合能力']= score_radar['收益能力']*0.30+\
    score_radar['风控能力']*0.25+ score_radar['风险调整绩效']*0.25 +score_radar['业绩持续性']*0.20#+#score_radar['持续性']*0.1
    star_level=[]
    for i in range(score_radar.size):
        if score_radar[i]>=90:  
                scar_level=5
        elif score_radar[i]<90 and score_radar[i]>=80:
                 scar_level=4.5
        elif score_radar[i]<80 and score_radar[i]>=70:
                 scar_level=4
        elif score_radar[i]<70 and score_radar[i]>=60:
                scar_level=3.5
        elif score_radar[i]>=50 and score_radar[i]<60:
                scar_level=3
        elif score_radar[i]>=40 and score_radar[i]<50:  
                scar_level=2.5
        else:
                scar_level=2          
        star_level.append(scar_level)           
    star_table=pd.Series(star_level,index=['收益能力','风控能力','风险调整绩效','业绩持续性','综合能力1'])  
    return score,score_radar, star_table
    
    
    
def det_bond(
        annret,
        annvol,
        sortino,
        Calmar,
        M2,
        signal_drawdown,
        winrate,
        Cvar95,
        zhibiao_strategy):
    #各项指标单项评分
     #胜率评分
#    if  win_ratio>=0.7:
#         score['胜率']=100     
#    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
#         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
#    elif win_ratio<=0.31:
#         score['胜率']=0
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['return_annual']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['索提诺比率']=((pd.Series(zhibiao_strategy['sortino']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
    score['卡玛比']=((pd.Series(zhibiao_strategy['calmar']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M2']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
    score['单一最大回撤']=((pd.Series(zhibiao_strategy['data_signaldrawdown']).astype(float)-signal_drawdown)>0).sum()/len(zhibiao_strategy)*100

#    score['净值连续上涨周数']=((pd.Series(zhibiao_strategy['净值上升最长周数']).astype(float)- up_week)<0).sum()/len(zhibiao_strategy)*100
#    score['净值连续下跌周数']=((pd.Series(zhibiao_strategy['净值下跌最长周数']).astype(float)- down_week)>0).sum()/len(zhibiao_strategy)*100
    score['胜率']=((pd.Series(zhibiao_strategy['Win_rate']).astype(float)-winrate)<0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['data_CVaR95']).astype(float)-Cvar95)>0).sum()/len(zhibiao_strategy)*100
    score['年化波动率']=((pd.Series(zhibiao_strategy['volatility_annual']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
    score_radar=pd.Series()
    score_radar['收益能力']= score['年化收益']
    score_radar['净值管理能力']=  score['单一最大回撤']
    score_radar['业绩持续性']= score['胜率']
  
    score_radar['风险调整绩效']= (score['索提诺比率']+score['卡玛比']+ score['M平方'])/3
    score_radar['风控能力']= (score['Cvar95']+ score['年化波动率'])/2
   
#    score_radar['行情判断能力']= score['胜率'] 
#    score_scar=pd.Series()
#    score_star=pd.Series()
    score_radar['综合能力']= score_radar['收益能力']*0.40+  score_radar['净值管理能力']*0.1+\
    score_radar['风控能力']*0.16+ score_radar['风险调整绩效']*0.24+score_radar['业绩持续性']*0.1#+#score_radar['持续性']*0.1
    #星级评定
    star_level=[]
    for i in range(score_radar.size):
        if score_radar[i]>=90:  
                scar_level=5
        elif score_radar[i]<90 and score_radar[i]>=80:
                 scar_level=4.5
        elif score_radar[i]<80 and score_radar[i]>=70:
                 scar_level=4
        elif score_radar[i]<70 and score_radar[i]>=60:
                scar_level=3.5
        elif score_radar[i]>=50 and score_radar[i]<60:
                scar_level=3
        elif score_radar[i]>=40 and score_radar[i]<50:  
                scar_level=2.5
        else:
                scar_level=2          
        star_level.append(scar_level)           
    star_table=pd.Series(star_level,index=['收益能力','净值管理能力','业绩持续性','风险调整绩效','风控能力','综合能力1']) 
#    star_table['相对收益能力']='--'
    return score,score_radar,star_table    
#def det_bond(
#        annret,
#        annvol,
#        sortino,
#        Calmar,
#        M2,
#        signal_drawdown,
#        winrate,
#        Cvar95,
#        zhibiao_strategy):
#    #各项指标单项评分
#     #胜率评分
##    if  win_ratio>=0.7:
##         score['胜率']=100     
##    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
##         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
##    elif win_ratio<=0.31:
##         score['胜率']=0
#    score=pd.Series()
#    score['年化收益']=((pd.Series(zhibiao_strategy['anyet']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
#    score['索提诺比率']=((pd.Series(zhibiao_strategy['索提诺']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
#    score['卡玛比']=((pd.Series(zhibiao_strategy['卡玛比']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
#    score['M平方']=((pd.Series(zhibiao_strategy['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
#    score['单一最大回撤']=((pd.Series(zhibiao_strategy['单一最大回撤']).astype(float)-signal_drawdown)>0).sum()/len(zhibiao_strategy)*100
#
##    score['净值连续上涨周数']=((pd.Series(zhibiao_strategy['净值上升最长周数']).astype(float)- up_week)<0).sum()/len(zhibiao_strategy)*100
##    score['净值连续下跌周数']=((pd.Series(zhibiao_strategy['净值下跌最长周数']).astype(float)- down_week)>0).sum()/len(zhibiao_strategy)*100
#    score['胜率']=((pd.Series(zhibiao_strategy['胜率']).astype(float)-winrate)<0).sum()/len(zhibiao_strategy)*100
#    score['Cvar95']=((pd.Series(zhibiao_strategy['CVaR(95%)']).astype(float)-Cvar95)>0).sum()/len(zhibiao_strategy)*100
#    score['年化波动率']=((pd.Series(zhibiao_strategy['年化波动率']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
#    score_radar=pd.Series()
#    score_radar['收益能力']= score['年化收益']
#    score_radar['净值管理能力']=  score['单一最大回撤']
#    score_radar['业绩持续性']= score['胜率']
#  
#    score_radar['风险调整绩效']= (score['索提诺比率']+score['卡玛比']+ score['M平方'])/3
#    score_radar['风控能力']= (score['Cvar95']+ score['年化波动率'])/2
#   
##    score_radar['行情判断能力']= score['胜率'] 
##    score_scar=pd.Series()
##    score_star=pd.Series()
#    score_radar['综合能力']= score_radar['收益能力']*0.40+  score_radar['净值管理能力']*0.1+\
#    score_radar['风控能力']*0.16+ score_radar['风险调整绩效']*0.24+score_radar['业绩持续性']*0.1#+#score_radar['持续性']*0.1
##    #星级评定
##    star_level=[]
##    for i in range(score_radar.size):
##        if score_radar[i]>=90:  
##                scar_level=5
##        elif score_radar[i]<90 and score_radar[i]>=80:
##                 scar_level=4.5
##        elif score_radar[i]<80 and score_radar[i]>=70:
##                 scar_level=4
##        elif score_radar[i]<70 and score_radar[i]>=60:
##                scar_level=3.5
##        elif score_radar[i]>=50 and score_radar[i]<60:
##                scar_level=3
##        elif score_radar[i]>=40 and score_radar[i]<50:  
##                scar_level=2.5
##        else:
##                scar_level=2          
##        star_level.append(scar_level)           
##    star_table=pd.Series(star_level,index=['收益能力','择时能力','择股能力','风控能力','风险调整绩效','业绩持续性','综合能力1']) 
##    star_table['相对收益能力']='--'
#    return score,score_radar

def det_cta(
        annret,
        sortino,
        Calmar,
        signal_drawdown,
        winrate,
        Cvar95,
        zhibiao_strategy):
    #各项指标单项评分
     #胜率评分
#    if  win_ratio>=0.7:
#         score['胜率']=100     
#    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
#         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
#    elif win_ratio<=0.31:
#         score['胜率']=0
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['anyet']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['索提诺比率']=((pd.Series(zhibiao_strategy['索提诺']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
    score['卡玛比']=((pd.Series(zhibiao_strategy['卡玛比']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
    score['单一最大回撤']=((pd.Series(zhibiao_strategy['单一最大回撤']).astype(float)-signal_drawdown)>0).sum()/len(zhibiao_strategy)*100

#    score['净值连续上涨周数']=((pd.Series(zhibiao_strategy['净值上升最长周数']).astype(float)- up_week)<0).sum()/len(zhibiao_strategy)*100
#    score['净值连续下跌周数']=((pd.Series(zhibiao_strategy['净值下跌最长周数']).astype(float)- down_week)>0).sum()/len(zhibiao_strategy)*100
    score['胜率']=((pd.Series(zhibiao_strategy['胜率']).astype(float)-winrate)<0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['CVaR(95%)']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
   
    score_radar=pd.Series()
    score_radar['收益能力']= score['年化收益']
    score_radar['净值管理能力']=  score['单一最大回撤']
    score_radar['业绩持续性']=score['胜率']
    score_radar['风险调整绩效']= (score['索提诺比率']+score['卡玛比'])/2
    score_radar['风控能力']=  score['Cvar95']
   
#    score_radar['行情判断能力']= score['胜率'] 
#    score_scar=pd.Series()
#    score_star=pd.Series()
    score_radar['综合能力']= score_radar['收益能力']*0.4+  score_radar['净值管理能力']*0.06+\
    score_radar['风控能力']*0.06+ score_radar['风险调整绩效']*0.35+score_radar['业绩持续性']*0.13#+#score_radar['持续性']*0.1
#    #星级评定
#    star_level=[]
#    for i in range(score_radar.size):
#        if score_radar[i]>=90:  
#                scar_level=5
#        elif score_radar[i]<90 and score_radar[i]>=80:
#                 scar_level=4.5
#        elif score_radar[i]<80 and score_radar[i]>=70:
#                 scar_level=4
#        elif score_radar[i]<70 and score_radar[i]>=60:
#                scar_level=3.5
#        elif score_radar[i]>=50 and score_radar[i]<60:
#                scar_level=3
#        elif score_radar[i]>=40 and score_radar[i]<50:  
#                scar_level=2.5
#        else:
#                scar_level=2          
#        star_level.append(scar_level)           
#    star_table=pd.Series(star_level,index=['收益能力','择时能力','择股能力','风控能力','风险调整绩效','业绩持续性','综合能力1']) 
#    star_table['相对收益能力']='--'
    return score,score_radar

def det_relative_macro(
        annret,
        sortino,
        Calmar,
        M2,
        signal_drawdown,
        winrate,
        Cvar95,
        zhibiao_strategy):
    #各项指标单项评分
     #胜率评分
#    if  win_ratio>=0.7:
#         score['胜率']=100     
#    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
#         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
#    elif win_ratio<=0.31:
#         score['胜率']=0
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['anyet']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
 
    score['索提诺比率']=((pd.Series(zhibiao_strategy['索提诺']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
    score['卡玛比']=((pd.Series(zhibiao_strategy['卡玛比']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
    score['单一最大回撤']=((pd.Series(zhibiao_strategy['单一最大回撤']).astype(float)-signal_drawdown)>0).sum()/len(zhibiao_strategy)*100

#    score['净值连续上涨周数']=((pd.Series(zhibiao_strategy['净值上升最长周数']).astype(float)- up_week)<0).sum()/len(zhibiao_strategy)*100
#    score['净值连续下跌周数']=((pd.Series(zhibiao_strategy['净值下跌最长周数']).astype(float)- down_week)>0).sum()/len(zhibiao_strategy)*100
    score['胜率']=((pd.Series(zhibiao_strategy['胜率']).astype(float)-winrate)<0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['CVaR(95%)']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
   
    score_radar=pd.Series()
    score_radar['收益能力']= score['年化收益']
    score_radar['净值管理能力']=  score['单一最大回撤']
    score_radar['业绩持续性']=score['胜率']
  
    score_radar['风险调整绩效']= (score['索提诺比率']+score['卡玛比']+ score['M平方'])/3
    score_radar['风控能力']=   score['Cvar95']
   
#    score_radar['行情判断能力']= score['胜率'] 
#    score_scar=pd.Series()
#    score_star=pd.Series()
    score_radar['综合能力']= score_radar['收益能力']*0.33+  score_radar['净值管理能力']*0.05+\
    score_radar['风控能力']*0.15+ score_radar['风险调整绩效']*0.32+score_radar['业绩持续性']*0.15#+#score_radar['持续性']*0.1
#    #星级评定
#    star_level=[]
#    for i in range(score_radar.size):
#        if score_radar[i]>=90:  
#                scar_level=5
#        elif score_radar[i]<90 and score_radar[i]>=80:
#                 scar_level=4.5
#        elif score_radar[i]<80 and score_radar[i]>=70:
#                 scar_level=4
#        elif score_radar[i]<70 and score_radar[i]>=60:
#                scar_level=3.5
#        elif score_radar[i]>=50 and score_radar[i]<60:
#                scar_level=3
#        elif score_radar[i]>=40 and score_radar[i]<50:  
#                scar_level=2.5
#        else:
#                scar_level=2          
#        star_level.append(scar_level)           
#    star_table=pd.Series(star_level,index=['收益能力','择时能力','择股能力','风控能力','风险调整绩效','业绩持续性','综合能力1']) 
#    star_table['相对收益能力']='--'
    return score,score_radar

#以下是公募基金评分
#混合型和QDII，QDII-ETF分级杠杆
def mixture_QDII_judge(annret,max_drawdown,Cvar95,M2,winrate,sharpratio,zhibiao_strategy):
    
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['anyet']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['最大回撤']=((pd.Series(zhibiao_strategy['最大回撤']).astype(float)-max_drawdown)>0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['CVaR(95%)']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
#    score['业绩一致性']=((pd.Series(zhibiao_strategy['扫描统计量']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['扫描统计量'].dropna().size)*100
    score['胜率']=((pd.Series(zhibiao_strategy['胜率']).astype(float)-winrate)<0).sum()/len(zhibiao_strategy)*100
    score['夏普比率']=((pd.Series(zhibiao_strategy['夏普比率']).astype(float)-sharpratio)<0).sum()/len(zhibiao_strategy)*100
    
    score_radar=pd.Series()
#    score_radar['收益能力']= score['年化收益']
#    score_radar['风控能力']= score['Cvar95']*0.50+score['最大回撤']*0.50
#    score_radar['风险调整收益']= score['M2']
#    score_radar['业绩持续性']= score['业绩一致性']
    score_radar['年化收益']=score['年化收益']
    score_radar['夏普比率']=score['夏普比率']
    score_radar['综合能力']= score['年化收益']*0.40+score['Cvar95']*0.15+score['最大回撤']*0.1+ score['M平方']*0.2+score['胜率']*0.15
    return score_radar
#债券型、定开债券等\理财型
def public_bond(annret,annvol,Cvar95,M2,winrate,sharpratio,zhibiao_strategy):
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['anyet']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['年化波动率']=((pd.Series(zhibiao_strategy['年化波动率']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['CVaR(95%)']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
#    score['业绩一致性']=((pd.Series(zhibiao_strategy['扫描统计量']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['扫描统计量'].dropna().size)*100
    score['胜率']=((pd.Series(zhibiao_strategy['胜率']).astype(float)-winrate)<0).sum()/len(zhibiao_strategy)*100
    score['夏普比率']=((pd.Series(zhibiao_strategy['夏普比率']).astype(float)-sharpratio)<0).sum()/len(zhibiao_strategy)*100
    
    score_radar=pd.Series()
#    score_radar['收益能力']= score['年化收益']
#    score_radar['风控能力']= score['Cvar95']*0.50+score['年化波动率']*0.50
#    score_radar['风险调整收益']= score['M2']
#    score_radar['业绩持续性']= score['业绩一致性']
    score_radar['年化收益']=score['年化收益']
    score_radar['夏普比率']=score['夏普比率']
    score_radar['综合能力']= score['年化收益']*0.4+score['Cvar95']*0.1+score['年化波动率']*0.1+ score['胜率']*0.2+score['M平方']*0.2
    
    return score_radar
#股票型
#def public_stock(annret,alpha_ret,annvol,Cvar95,TM_alpha,TM_beta2,M2,inforatio,saomiao,sharpratio,zhibiao_strategy):
#    score=pd.Series()
#    score['年化收益']=((pd.Series(zhibiao_strategy['anyet']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
#    score['超额收益']=((pd.Series(zhibiao_strategy['阿尔法']).astype(float)-alpha_ret)<0).sum()/len(zhibiao_strategy)*100   
#    score['年化波动率']=((pd.Series(zhibiao_strategy['年化波动率']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
#    score['Cvar95']=((pd.Series(zhibiao_strategy['CVaR(95%)']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
#   
#    score['TM择股能力']=((pd.Series(zhibiao_strategy['TM阿尔法']).astype(float)- TM_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['TM择时能力']=((pd.Series(zhibiao_strategy['TM_beta2']).astype(float)- TM_beta2)<0).sum()/len(zhibiao_strategy)*100
#    score['夏普比率']=((pd.Series(zhibiao_strategy['夏普比率']).astype(float)-sharpratio)<0).sum()/len(zhibiao_strategy)*100
#    
#    score['M平方']=((pd.Series(zhibiao_strategy['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
#    score['信息比率']=((pd.Series(zhibiao_strategy['信息比率']).astype(float)-inforatio)<0).sum()/len(zhibiao_strategy)*100
#    score['业绩一致性']=((pd.Series(zhibiao_strategy['扫描统计量']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['扫描统计量'].dropna().size)*100
#    score_radar=pd.Series()
##    score_radar['收益能力']= score['年化收益']*0.5+ score['超额收益']*0.5
##    score_radar['风控能力']= score['Cvar95']*0.50+score['年化波动率']*0.50
##    score_radar['资产管理能力']= score['TM择股能力']*0.5+score['TM择时能力']*0.5
##    score_radar['风险调整收益']= score['M2']*0.5+score['信息比率']*0.5
##    score_radar['业绩持续性']= score['业绩一致性']
#    score_radar['年化收益']=score['年化收益']
#    score_radar['夏普比率']=score['夏普比率']
#    score_radar['综合能力']= score['年化收益']*0.1+ score['超额收益']*0.1+score['Cvar95']*0.1+score['年化波动率']*0.1+score['TM择股能力']*0.1+score['TM择时能力']*0.1+ score['M平方']*0.1+score['信息比率']*0.1+ score['业绩一致性']*0.20
#    return score_radar
def public_stock(
        annret,
        alpha_ret,
        sortino,
        Calmar,
        M2,
        signal_drawdown,
        winrate,
        Cvar95,
        zhibiao_strategy):
    #各项指标单项评分
     #胜率评分
#    if  win_ratio>=0.7:
#         score['胜率']=100     
#    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
#         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
#    elif win_ratio<=0.31:
#         score['胜率']=0
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['anyet']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['超额收益']=((pd.Series(zhibiao_strategy['阿尔法']).astype(float)-alpha_ret)<0).sum()/len(zhibiao_strategy)*100
    score['索提诺比率']=((pd.Series(zhibiao_strategy['索提诺']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
    score['卡玛比']=((pd.Series(zhibiao_strategy['卡玛比']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
    score['单一最大回撤']=((pd.Series(zhibiao_strategy['单一最大回撤']).astype(float)-signal_drawdown)>0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['CVaR(95%)']).astype(float)-Cvar95)>0).sum()/len(zhibiao_strategy)*100
   
#    score['净值连续上涨周数']=((pd.Series(zhibiao_strategy['净值上升最长周数']).astype(float)- up_week)<0).sum()/len(zhibiao_strategy)*100
#    score['净值连续下跌周数']=((pd.Series(zhibiao_strategy['净值下跌最长周数']).astype(float)- down_week)>0).sum()/len(zhibiao_strategy)*100
    score['胜率']=((pd.Series(zhibiao_strategy['胜率']).astype(float)-winrate)<0).sum()/len(zhibiao_strategy)*100
#    score['市场上涨区间收益']=((pd.Series(zhibiao_strategy['产品收益(市场上涨)']).astype(float)- ret_upmarket)<0).sum()/len(zhibiao_strategy)*100
#    score['市场下跌区间收益']=((pd.Series(zhibiao_strategy['产品收益(市场下跌)']).astype(float)- ret_downmarket)<0).sum()/len(zhibiao_strategy)*100
#    
   
    score_radar=pd.Series()
    score_radar['收益能力']= score['年化收益']
    score_radar['净值管理能力']=  score['单一最大回撤']
    score_radar['业绩持续性']= score['胜率']
#    score_radar['盈利能力']=score['市场上涨区间收益']
    score_radar['风险调整绩效']= (score['索提诺比率']+score['卡玛比']+ score['M平方'])/3
    score_radar['风控能力']=  score['Cvar95']
    score_radar['超额收益能力']=   score['超额收益']
    
#    score_radar['行情判断能力']= score['胜率'] 
#    score_scar=pd.Series()
#    score_star=pd.Series()
    score_radar['综合能力']= score_radar['收益能力']*0.1+  score_radar['净值管理能力']*0.06+\
    score_radar['风控能力']*0.1+ score_radar['风险调整绩效']*0.38 +score_radar['业绩持续性']*0.1+ score_radar['超额收益能力']*0.26#+#score_radar['持续性']*0.1
#    #星级评定
#    star_level=[]
#    for i in range(score_radar.size):
#        if score_radar[i]>=90:  
#                scar_level=5
#        elif score_radar[i]<90 and score_radar[i]>=80:
#                 scar_level=4.5
#        elif score_radar[i]<80 and score_radar[i]>=70:
#                 scar_level=4
#        elif score_radar[i]<70 and score_radar[i]>=60:
#                scar_level=3.5
#        elif score_radar[i]>=50 and score_radar[i]<60:
#                scar_level=3
#        elif score_radar[i]>=40 and score_radar[i]<50:  
#                scar_level=2.5
#        else:
#                scar_level=2          
#        star_level.append(scar_level)           
#    star_table=pd.Series(star_level,index=['收益能力','择时能力','择股能力','风控能力','风险调整绩效','业绩持续性','综合能力1']) 
#    star_table['相对收益能力']='--'
    return score,score_radar



#货币型还缺少影价偏离度绝对值的平均值\
def public_money(annret,money_annvol,Cvar95,M2,winrate,sharpratio,zhibiao_strategy):
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['anyet']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100 
    score['年化波动率']=((pd.Series(zhibiao_strategy['年化波动率']).astype(float)-money_annvol)>0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['CVaR(95%)']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
   
    score['M平方']=((pd.Series(zhibiao_strategy['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
#    score['业绩一致性']=((pd.Series(zhibiao_strategy['扫描统计量']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['扫描统计量'].dropna().size)*100
    score['夏普比率']=((pd.Series(zhibiao_strategy['夏普比率']).astype(float)-sharpratio)<0).sum()/len(zhibiao_strategy)*100
    score['胜率']=((pd.Series(zhibiao_strategy['胜率']).astype(float)-winrate)<0).sum()/len(zhibiao_strategy)*100
    score_radar=pd.Series()
#    score_radar['收益能力']= score['年化收益']
#    score_radar['风控能力']= score['年化波动率']*0.50+score['Cvar95']*0.5
#    score_radar['风险调整收益']= score['M2']
#    score_radar['业绩持续性']= score['业绩一致性']
    score_radar['年化收益']=score['年化收益']
    score_radar['夏普比率']=score['夏普比率']
    score_radar['综合能力']= score['年化收益']*0.4+ score['胜率']*0.2+score['年化波动率']*0.1+ score['M平方']*0.2
    return score_radar
#指数型跟踪误差和信息比率没有用自身的基准
def public_stock_index(annret,std_div,inforatio,M2,manage_fee,trustee_fee,sharpratio,zhibiao_strategy):
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['anyet']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100 
    score['跟踪误差']=((pd.Series(zhibiao_strategy['跟踪误差']).astype(float)-std_div)>0).sum()/len(zhibiao_strategy)*100 
    score['信息比率']=((pd.Series(zhibiao_strategy['信息比率']).astype(float)-inforatio)<0).sum()/len(zhibiao_strategy)*100
    score['manage_fee']=((pd.Series(zhibiao_strategy['管理费']).astype(float)-manage_fee)>0).sum()/len(zhibiao_strategy)*100
    score['trustee_fee']=((pd.Series(zhibiao_strategy['托管费']).astype(float)-trustee_fee)>0).sum()/len(zhibiao_strategy)*100
    score['夏普比率']=((pd.Series(zhibiao_strategy['夏普比率']).astype(float)-sharpratio)<0).sum()/len(zhibiao_strategy)*100
    
    score['M平方']=((pd.Series(zhibiao_strategy['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
    score_radar=pd.Series()
    score_radar['年化收益']=score['年化收益']
    score_radar['夏普比率']=score['夏普比率']
    score_radar['综合能力']= score['跟踪误差']*0.45+ score['信息比率']*0.15+score['manage_fee']*0.1+ score['trustee_fee']*0.1+score['M平方']*0.2
    return score_radar


#def ranking(strategy_type,return_annual,drawback_max,volatility_annual,sharpe,Calmar):
#    """
#    产品打分
#    """
#    score = 0
#    for i in range()
#    if strategy_type in ('宏观策略' , '宏观对冲' , '全球宏观','复合策略'):
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
#    elif strategy_type in ('相对价值' , '市场中性' , '量化对冲' , '股票对冲'):
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.1, 0, -0.01, -0.03, 0.07, 0.13)
#    elif strategy_type in ('管理期货' , 'CTA策略' , 'CTA'):
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.15, 0.05, -0.03, -0.07, 0.1, 0.16)
#    elif strategy_type in ('债券策略' , '债券型','固定收益'):
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.08, 0.055, -0.005, -0.015, 0.03, 0.07)
#    elif strategy_type in ('股票多头' , '股票型' , '股票策略' , '股票量化'):
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
#    else:
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
#    return score

def zhibiao_description(annyet,alpha_yet,annVol, mean_div,r2_risk,div,beta,beta_niu,beta_xiong,updown_ratio,downrisk,maxdrawdown,meandrawdown,signal_maxdrawdown,VARRatio95,CVar95,VARRatio99,CVar99,\
                    alpha,Sharp,M2,Calmar,Stutzer,inforatio,Sterling,omega,burke,Sortino,treynor,HM_alpha,TM_alpha,CL_alpha,beta_ratio,CL_beta,HM_beta,TM_beta,zhibiao_strategy):
     

    if 'return_annual' in zhibiao_strategy.columns.tolist():
        zhibiao_miaoshu=pd.Series()
        if annyet<zhibiao_strategy['return_annual'].quantile(0.50): 
           zhibiao_miaoshu['年化收益率']='一般'
        elif annyet>=zhibiao_strategy['return_annual'].quantile(0.50) and  annyet<zhibiao_strategy['return_annual'].quantile(0.75):
           zhibiao_miaoshu['年化收益率']='良好'
        elif annyet>=zhibiao_strategy['return_annual'].quantile(0.75) :
          zhibiao_miaoshu['年化收益率']='优秀'
       
        #超额收益
        if alpha_yet<zhibiao_strategy['alpha'].quantile(0.50): 
           zhibiao_miaoshu['alpha(年化)']='一般'
        elif alpha_yet>=zhibiao_strategy['alpha'].quantile(0.50) and  alpha_yet<zhibiao_strategy['alpha'].quantile(0.75):
           zhibiao_miaoshu['alpha(年化)']='良好'
        elif alpha_yet>=zhibiao_strategy['alpha'].quantile(0.75) :
          zhibiao_miaoshu['alpha(年化)']='优秀'

       #日收益率波动率
#        if dayVol<zhibiao_strategy['年化波动率'].quantile(0.50):
#           zhibiao_miaoshu['年化波动率']='较小'
#        elif dayVol>=zhibiao_strategy['年化波动率'].quantile(0.50) and dayVol<zhibiao_strategy['年化波动率'].quantile(0.75):
#          zhibiao_miaoshu['年化波动率']='较大'
#        elif dayVol>=zhibiao_strategy['年化波动率'].quantile(0.75) :
#          zhibiao_miaoshu['年化波动率']='很大' 
        
        #年化波动性
        if annVol<zhibiao_strategy['volatility_annual'].quantile(0.50):
           zhibiao_miaoshu['年化波动率']='优秀'
        elif annVol>=zhibiao_strategy['volatility_annual'].quantile(0.50) and annVol<zhibiao_strategy['volatility_annual'].quantile(0.75):
          zhibiao_miaoshu['年化波动率']='良好'
        elif annVol>=zhibiao_strategy['volatility_annual'].quantile(0.75) :
          zhibiao_miaoshu['年化波动率']='一般'
    
     #beta    
        if beta<zhibiao_strategy['beta'].quantile(0.50):
           zhibiao_miaoshu['beta']='--'
        elif beta>=zhibiao_strategy['beta'].quantile(0.50) and  beta<zhibiao_strategy['beta'].quantile(0.75):
          zhibiao_miaoshu['beta']='--'
        elif beta>=zhibiao_strategy['beta'].quantile(0.75) :
           zhibiao_miaoshu['beta']='--'
           
        #上下行比率
        if updown_ratio<zhibiao_strategy['updownratio'].astype(float).quantile(0.50):
           zhibiao_miaoshu['上下行比率']='一般'
        elif  updown_ratio>=zhibiao_strategy['updownratio'].astype(float).quantile(0.50) and  updown_ratio<zhibiao_strategy['updownratio'].astype(float).quantile(0.75) and  updown_ratio>1:
          zhibiao_miaoshu['上下行比率']='良好'
        elif  updown_ratio>=zhibiao_strategy['updownratio'].astype(float).quantile(0.75) and  updown_ratio>1 :
           zhibiao_miaoshu['上下行比率']='优秀'
        else:
           zhibiao_miaoshu['上下行比率']='一般'  
        
        #跟踪误差
        if div<zhibiao_strategy['Std_devi'].astype(float).quantile(0.50):
           zhibiao_miaoshu['跟踪误差']='优秀'
        elif  div>=zhibiao_strategy['Std_devi'].astype(float).quantile(0.50) and div<zhibiao_strategy['Std_devi'].astype(float).quantile(0.75) :
          zhibiao_miaoshu['跟踪误差']='良好'
        elif  div>=zhibiao_strategy['Std_devi'].astype(float).quantile(0.75)  :
           zhibiao_miaoshu['跟踪误差']='一般'
     
        #残差风险
        if r2_risk<zhibiao_strategy['R2_risk'].astype(float).quantile(0.50):
           zhibiao_miaoshu['残差风险']='优秀'
        elif  r2_risk>=zhibiao_strategy['R2_risk'].astype(float).quantile(0.50) and  r2_risk<zhibiao_strategy['R2_risk'].astype(float).quantile(0.75) :
          zhibiao_miaoshu['残差风险']='良好'
        elif  r2_risk>=zhibiao_strategy['R2_risk'].astype(float).quantile(0.75)  :
           zhibiao_miaoshu['残差风险']='一般'
      
        #牛市贝塔
        if beta_niu<zhibiao_strategy['Beta_niu'].quantile(0.50):
           zhibiao_miaoshu['牛市beta']='--'
        elif  beta_niu>=zhibiao_strategy['Beta_niu'].quantile(0.50) and  beta_niu<zhibiao_strategy['Beta_niu'].quantile(0.75) and  beta_niu>1:
          zhibiao_miaoshu['牛市beta']='--'
        elif  beta_niu>=zhibiao_strategy['Beta_niu'].quantile(0.75) and beta_niu>1 :
           zhibiao_miaoshu['牛市beta']='--'
        else:
           zhibiao_miaoshu['牛市beta']='--' 
        
          #熊市贝塔
        if beta_xiong<zhibiao_strategy['Beta_xiong'].quantile(0.50):
           zhibiao_miaoshu['熊市beta']='--'
        elif  beta_xiong>=zhibiao_strategy['Beta_xiong'].quantile(0.50) and  beta_xiong<zhibiao_strategy['Beta_xiong'].quantile(0.75) :
          zhibiao_miaoshu['熊市beta']='--'
        elif  beta_xiong>=zhibiao_strategy['Beta_xiong'].quantile(0.75):
           zhibiao_miaoshu['熊市beta']='--'
       
        #平均绝对偏差
        if mean_div<zhibiao_strategy['mean_devi'].astype(float).quantile(0.50):
           zhibiao_miaoshu['平均绝对偏差']='优秀'
        elif   mean_div>=zhibiao_strategy['mean_devi'].astype(float).quantile(0.50) and   mean_div<zhibiao_strategy['mean_devi'].astype(float).quantile(0.75) :
          zhibiao_miaoshu['平均绝对偏差']='良好'
        elif   mean_div>=zhibiao_strategy['mean_devi'].astype(float).quantile(0.75) :
           zhibiao_miaoshu['平均绝对偏差']='一般'
        
        #下行风险
        if downrisk<zhibiao_strategy['downrisk'].quantile(0.50):
           zhibiao_miaoshu['下行风险']='优秀'
        elif downrisk>=zhibiao_strategy['downrisk'].quantile(0.50) and  downrisk<zhibiao_strategy['downrisk'].quantile(0.75):
          zhibiao_miaoshu['下行风险']='良好'
        elif downrisk>=zhibiao_strategy['downrisk'].quantile(0.75) :
          zhibiao_miaoshu['下行风险']='一般'
          
            #最大回撤
        if maxdrawdown<zhibiao_strategy['drawback_max'].quantile(0.50):
           zhibiao_miaoshu['最大回撤']='优秀'
        elif maxdrawdown>=zhibiao_strategy['drawback_max'].quantile(0.50) and  maxdrawdown<zhibiao_strategy['drawback_max'].quantile(0.75):
           zhibiao_miaoshu['最大回撤']='良好'
        elif maxdrawdown>=zhibiao_strategy['drawback_max'].quantile(0.75):
           zhibiao_miaoshu['最大回撤']='一般'
   
        #平均回撤
        if meandrawdown<zhibiao_strategy['data_mean_drawdown'].astype(float).quantile(0.50):
           zhibiao_miaoshu['平均回撤']='优秀'
        elif meandrawdown>=zhibiao_strategy['data_mean_drawdown'].quantile(0.50) and  meandrawdown<zhibiao_strategy['data_mean_drawdown'].quantile(0.75):
           zhibiao_miaoshu['平均回撤']='良好'
        elif meandrawdown>=zhibiao_strategy['data_mean_drawdown'].quantile(0.75):
           zhibiao_miaoshu['平均回撤']='一般'
            #最大单一回撤  
        if signal_maxdrawdown<zhibiao_strategy['data_signaldrawdown'].quantile(0.50):
          zhibiao_miaoshu['最大单一回撤']='优秀'
        elif signal_maxdrawdown>=zhibiao_strategy['data_signaldrawdown'].quantile(0.50) and  signal_maxdrawdown<zhibiao_strategy['data_signaldrawdown'].quantile(0.75):
          zhibiao_miaoshu['最大单一回撤']='良好'
        elif signal_maxdrawdown>=zhibiao_strategy['data_signaldrawdown'].quantile(0.75) :
          zhibiao_miaoshu['最大单一回撤']='一般'
  
            #Var
        if VARRatio95<zhibiao_strategy['data_VARRatio95'].quantile(0.50):
           zhibiao_miaoshu['VaR(95%/10天)']='一般'
        elif VARRatio95>=zhibiao_strategy['data_VARRatio95'].quantile(0.50) and  VARRatio95<zhibiao_strategy['data_VARRatio95'].quantile(0.75):
           zhibiao_miaoshu['VaR(95%/10天)']='良好'
        elif VARRatio95>=zhibiao_strategy['data_VARRatio95'].quantile(0.75) :
          zhibiao_miaoshu['VaR(95%/10天)']='优秀'
   
        
             #CVar
        if CVar95<zhibiao_strategy['data_CVaR95'].quantile(0.50):
           zhibiao_miaoshu['CVaR(95%/10天)']='一般'
        elif CVar95>=zhibiao_strategy['data_CVaR95'].quantile(0.50) and  CVar95<zhibiao_strategy['data_CVaR95'].quantile(0.75):
          zhibiao_miaoshu['CVaR(95%/10天)']='良好'
        elif CVar95>=zhibiao_strategy['data_CVaR95'].quantile(0.75):
          zhibiao_miaoshu['CVaR(95%/10天)']='优秀'
            
             #Var
        if VARRatio99<zhibiao_strategy['data_VARRatio99'].quantile(0.50):
           zhibiao_miaoshu['VaR(99%)']='一般'
        elif VARRatio99>=zhibiao_strategy['data_VARRatio99'].quantile(0.50) and  VARRatio99<zhibiao_strategy['data_VARRatio99'].quantile(0.75):
           zhibiao_miaoshu['VaR(99%)']='良好'
        elif VARRatio99>=zhibiao_strategy['data_VARRatio99'].quantile(0.75) :
          zhibiao_miaoshu['VaR(99%)']='优秀'
   
        
             #CVar
        if CVar99<zhibiao_strategy['data_CVaR99'].quantile(0.50):
           zhibiao_miaoshu['CVaR(99%)']='一般'
        elif CVar99>=zhibiao_strategy['data_CVaR99'].quantile(0.50) and  CVar99<zhibiao_strategy['data_CVaR99'].quantile(0.75):
          zhibiao_miaoshu['CVaR(99%)']='良好'
        elif CVar99>=zhibiao_strategy['data_CVaR99'].quantile(0.75):
          zhibiao_miaoshu['CVaR(99%)']='优秀'
        
        
        
#        超额收益alpha
        
        if alpha<zhibiao_strategy['alpha'].quantile(0.50):
           zhibiao_miaoshu['超额收益']='一般'
        elif alpha>=zhibiao_strategy['alpha'].quantile(0.50) and  alpha<zhibiao_strategy['alpha'].quantile(0.75) and alpha>0:
          zhibiao_miaoshu['超额收益']='良好'
        elif alpha>=zhibiao_strategy['alpha'].quantile(0.75) and alpha>0:
          zhibiao_miaoshu['超额收益']='优秀'
        else:
           zhibiao_miaoshu['超额收益']='一般'
#        
        #夏普比率
        if Sharp<zhibiao_strategy['sharpe'].quantile(0.50):
           zhibiao_miaoshu['夏普比率']='一般'
        elif Sharp>=zhibiao_strategy['sharpe'].quantile(0.50) and  Sharp<zhibiao_strategy['sharpe'].quantile(0.75) and  Sharp>0:
           zhibiao_miaoshu['夏普比率']='良好'
        elif Sharp>=zhibiao_strategy['sharpe'].quantile(0.75)  and Sharp>0:
           zhibiao_miaoshu['夏普比率']='优秀'
        else:
           zhibiao_miaoshu['夏普比率']='一般'
        
        #M平方
        if M2<zhibiao_strategy['M2'].quantile(0.50):
           zhibiao_miaoshu['M平方']='一般'
        elif M2>=zhibiao_strategy['M2'].quantile(0.50) and  M2<zhibiao_strategy['M2'].quantile(0.75):
           zhibiao_miaoshu['M平方']='良好'
        elif M2>=zhibiao_strategy['M2'].quantile(0.75) : 
           zhibiao_miaoshu['M平方']='优秀'
        
        #calmar
        if Calmar<zhibiao_strategy['calmar'].quantile(0.50):
           zhibiao_miaoshu['卡玛比率']='一般'
        elif Calmar>=zhibiao_strategy['calmar'].quantile(0.50) and  Calmar<zhibiao_strategy['calmar'].quantile(0.75):
           zhibiao_miaoshu['卡玛比率']='良好'
        elif Calmar>=zhibiao_strategy['calmar'].quantile(0.75): 
          zhibiao_miaoshu['卡玛比率']='优秀'
    
    #斯图泽指数    
   
        if Stutzer<zhibiao_strategy['stutzer'].astype(float).quantile(0.50):
           zhibiao_miaoshu['斯图泽指数']='一般'
        elif Stutzer>=zhibiao_strategy['stutzer'].astype(float).quantile(0.50) and   Stutzer<zhibiao_strategy['stutzer'].astype(float).quantile(0.75) and Stutzer>0:
           zhibiao_miaoshu['斯图泽指数']='良好'
        elif Stutzer>=zhibiao_strategy['stutzer'].astype(float).quantile(0.75) and Stutzer>0: 
          zhibiao_miaoshu['斯图泽指数']='优秀'
        else:
          zhibiao_miaoshu['斯图泽指数']='一般'
        
        #信息比率
        if inforatio<zhibiao_strategy['information_ratio'].astype(float).quantile(0.50):
           zhibiao_miaoshu['信息比率']='一般'
        elif inforatio>=zhibiao_strategy['information_ratio'].astype(float).quantile(0.50)  and inforatio<zhibiao_strategy['information_ratio'].astype(float).quantile(0.75) and  inforatio>0:
           zhibiao_miaoshu['信息比率']='良好'
        elif inforatio>=zhibiao_strategy['information_ratio'].astype(float).quantile(0.75)  and  inforatio>0: 
           zhibiao_miaoshu['信息比率']='优秀'
        else:
           zhibiao_miaoshu['信息比率']='一般'
       
        #sterling
        if Sterling<zhibiao_strategy['sterling_all'].astype(float).quantile(0.50):
          zhibiao_miaoshu['Sterling指数']='一般'
        elif Sterling>=zhibiao_strategy['sterling_all'].astype(float).quantile(0.50) and Sterling<zhibiao_strategy['sterling_all'].astype(float).quantile(0.75) and  Sterling>0:
          zhibiao_miaoshu['Sterling指数']='良好'
        elif Sterling>=zhibiao_strategy['sterling_all'].astype(float).quantile(0.75) and  Sterling>0: 
          zhibiao_miaoshu['Sterling指数']='优秀'
        else:
           zhibiao_miaoshu['Sterling指数']='一般'
       
        #omega指数
        if omega<zhibiao_strategy['omega'].astype(float).quantile(0.50):
          zhibiao_miaoshu['Omega比率']='一般'
        elif omega>=zhibiao_strategy['omega'].astype(float).quantile(0.50) and  omega<zhibiao_strategy['omega'].astype(float).quantile(0.75) :
           zhibiao_miaoshu['Omega比率']='良好'
        elif omega>=zhibiao_strategy['omega'].astype(float).quantile(0.75) : 
           zhibiao_miaoshu['Omega比率']='优秀'
   
        #burke比率
        if burke<zhibiao_strategy['Burke_ratio_all'].astype(float).quantile(0.50):
          zhibiao_miaoshu['Burke比率']='一般'
        elif burke>=zhibiao_strategy['Burke_ratio_all'].astype(float).quantile(0.50) and  burke<zhibiao_strategy['Burke_ratio_all'].astype(float).quantile(0.75) and burke>0:
           zhibiao_miaoshu['Burke比率']='良好'
        elif burke>=zhibiao_strategy['Burke_ratio_all'].astype(float).quantile(0.75) and burke>0: 
          zhibiao_miaoshu['Burke比率']='优秀'
        else:
          zhibiao_miaoshu['Burke比率']='一般'
        
        #索提诺比率
        if Sortino<zhibiao_strategy['sortino'].astype(float).quantile(0.50):
           zhibiao_miaoshu['索提诺比率']='一般'
        elif Sortino>=zhibiao_strategy['sortino'].astype(float).quantile(0.50) and  Sortino<zhibiao_strategy['sortino'].astype(float).quantile(0.75) and Sortino>0:
          zhibiao_miaoshu['索提诺比率']='良好'
        elif Sortino>=zhibiao_strategy['sortino'].astype(float).quantile(0.75) and Sortino>0: 
           zhibiao_miaoshu['索提诺比率']='优秀'
        else:
           zhibiao_miaoshu['索提诺比率']='一般'
        treynor
          #索提诺比率
        if treynor<zhibiao_strategy['treynor'].astype(float).quantile(0.50):
           zhibiao_miaoshu['特雷诺比率']='一般'
        elif treynor>=zhibiao_strategy['treynor'].astype(float).quantile(0.50) and  treynor<zhibiao_strategy['treynor'].astype(float).quantile(0.75):
          zhibiao_miaoshu['特雷诺比率']='良好'
        elif treynor>=zhibiao_strategy['treynor'].astype(float).quantile(0.75) : 
           zhibiao_miaoshu['特雷诺比率']='优秀'
    
        
        #hm择股
        if HM_alpha<zhibiao_strategy['HM_alpha'].astype(float).quantile(0.50):
           zhibiao_miaoshu['HM择股能力']='一般'
        elif HM_alpha>=zhibiao_strategy['HM_alpha'].astype(float).quantile(0.50) and  HM_alpha<zhibiao_strategy['HM_alpha'].astype(float).quantile(0.75) and HM_alpha>0:
          zhibiao_miaoshu['HM择股能力']='良好'
        elif HM_alpha>=zhibiao_strategy['HM_alpha'].astype(float).quantile(0.75) and HM_alpha>0: 
           zhibiao_miaoshu['HM择股能力']='优秀'
        else:
           zhibiao_miaoshu['HM择股能力']='一般'
        #tm择股
        if TM_alpha<zhibiao_strategy['TM_alpha'].astype(float).quantile(0.50):
           zhibiao_miaoshu['TM择股能力']='一般'
        elif TM_alpha>=zhibiao_strategy['TM_alpha'].astype(float).quantile(0.50) and  TM_alpha<zhibiao_strategy['TM_alpha'].astype(float).quantile(0.75) and TM_alpha>0:
          zhibiao_miaoshu['TM择股能力']='良好'
        elif TM_alpha>=zhibiao_strategy['TM_alpha'].astype(float).quantile(0.75) and TM_alpha>0: 
           zhibiao_miaoshu['TM择股能力']='优秀'
        else:
           zhibiao_miaoshu['TM择股能力']='一般'
        
          #CL择股
        if CL_alpha<zhibiao_strategy['CL_alpha'].astype(float).quantile(0.50):
           zhibiao_miaoshu['CL择股能力']='一般'
        elif CL_alpha>=zhibiao_strategy['CL_alpha'].astype(float).quantile(0.50) and  CL_alpha<zhibiao_strategy['CL_alpha'].astype(float).quantile(0.75) and CL_alpha>0:
          zhibiao_miaoshu['CL择股能力']='良好'
        elif CL_alpha>=zhibiao_strategy['CL_alpha'].astype(float).quantile(0.75) and CL_alpha>0: 
           zhibiao_miaoshu['CL择股能力']='优秀'
        else:
           zhibiao_miaoshu['CL择股能力']='一般'
        #贝塔择时比率
        if beta_ratio<zhibiao_strategy['beita_zeshi_all'].astype(float).quantile(0.50):
           zhibiao_miaoshu['贝塔择时比率']='一般'
        elif beta_ratio>=zhibiao_strategy['beita_zeshi_all'].astype(float).quantile(0.50) and beta_ratio<zhibiao_strategy['beita_zeshi_all'].astype(float).quantile(0.75) and beta_ratio>0:
          zhibiao_miaoshu['贝塔择时比率']='良好'
        elif beta_ratio>=zhibiao_strategy['beita_zeshi_all'].astype(float).quantile(0.75) and beta_ratio>0: 
           zhibiao_miaoshu['贝塔择时比率']='优秀'
        else:
           zhibiao_miaoshu['贝塔择时比率']='一般'
        #CL择时能力
        if CL_beta<zhibiao_strategy['CL_beta_ratio'].astype(float).quantile(0.50):
           zhibiao_miaoshu['CL择时能力']='一般'
        elif  CL_beta>=zhibiao_strategy['CL_beta_ratio'].astype(float).quantile(0.50) and   CL_beta<zhibiao_strategy['CL_beta_ratio'].astype(float).quantile(0.75) and  CL_beta>0:
          zhibiao_miaoshu['CL择时能力']='良好'
        elif  CL_beta>=zhibiao_strategy['CL_beta_ratio'].astype(float).quantile(0.75) and  CL_beta>0: 
           zhibiao_miaoshu['CL择时能力']='优秀'
        else:
           zhibiao_miaoshu['CL择时能力']='一般'
        
         #HM择时能力
        if HM_beta<zhibiao_strategy['HM_beta2'].astype(float).quantile(0.50):
           zhibiao_miaoshu['HM择时能力']='一般'
        elif  HM_beta>=zhibiao_strategy['HM_beta2'].astype(float).quantile(0.50) and   HM_beta<zhibiao_strategy['HM_beta2'].astype(float).quantile(0.75) and HM_beta>0:
          zhibiao_miaoshu['HM择时能力']='良好'
        elif  HM_beta>=zhibiao_strategy['HM_beta2'].astype(float).quantile(0.75) and HM_beta>0: 
           zhibiao_miaoshu['HM择时能力']='优秀'
        else:
           zhibiao_miaoshu['HM择时能力']='一般'
        
          #HM择时能力
        if TM_beta<zhibiao_strategy['TM_beta2'].astype(float).quantile(0.50):
           zhibiao_miaoshu['TM择时能力']='一般'
        elif  TM_beta>=zhibiao_strategy['TM_beta2'].astype(float).quantile(0.50) and  TM_beta<zhibiao_strategy['TM_beta2'].astype(float).quantile(0.75) and TM_beta>0:
          zhibiao_miaoshu['TM择时能力']='良好'
        elif  TM_beta>=zhibiao_strategy['TM_beta2'].astype(float).quantile(0.75) and  TM_beta>0: 
           zhibiao_miaoshu['TM择时能力']='优秀'
        else:
           zhibiao_miaoshu['TM择时能力']='一般'
        
        
        return zhibiao_miaoshu            
    else:
        #年化收益率
            zhibiao_miaoshu=pd.Series()
            if annyet<zhibiao_strategy['年化收益'].quantile(0.50): 
               zhibiao_miaoshu['年化收益率']='一般'
            elif annyet>=zhibiao_strategy['年化收益'].quantile(0.50) and  annyet<zhibiao_strategy['年化收益'].quantile(0.75):
               zhibiao_miaoshu['年化收益率']='良好'
            elif annyet>=zhibiao_strategy['年化收益'].quantile(0.75) :
              zhibiao_miaoshu['年化收益率']='优秀'
           
    #        #超额收益
            if alpha_yet<zhibiao_strategy['超额收益'].quantile(0.50): 
               zhibiao_miaoshu['alpha(年化)']='一般'
            elif alpha_yet>=zhibiao_strategy['超额收益'].quantile(0.50) and  alpha_yet<zhibiao_strategy['超额收益'].quantile(0.75):
               zhibiao_miaoshu['alpha(年化)']='良好'
            elif alpha_yet>=zhibiao_strategy['超额收益'].quantile(0.75) :
              zhibiao_miaoshu['alpha(年化)']='优秀'
    
           #日收益率波动率
    #        if dayVol<zhibiao_strategy['年化波动率'].quantile(0.50):
    #           zhibiao_miaoshu['年化波动率']='较小'
    #        elif dayVol>=zhibiao_strategy['年化波动率'].quantile(0.50) and dayVol<zhibiao_strategy['年化波动率'].quantile(0.75):
    #          zhibiao_miaoshu['年化波动率']='较大'
    #        elif dayVol>=zhibiao_strategy['年化波动率'].quantile(0.75) :
    #          zhibiao_miaoshu['年化波动率']='很大' 
            
            #年化波动性
            if annVol<zhibiao_strategy['年化波动率'].quantile(0.50):
               zhibiao_miaoshu['年化波动率']='优秀'
            elif annVol>=zhibiao_strategy['年化波动率'].quantile(0.50) and annVol<zhibiao_strategy['年化波动率'].quantile(0.75):
              zhibiao_miaoshu['年化波动率']='良好'
            elif annVol>=zhibiao_strategy['年化波动率'].quantile(0.75) :
              zhibiao_miaoshu['年化波动率']='一般'
        
         #beta    
            if beta<zhibiao_strategy['beta'].quantile(0.50):
               zhibiao_miaoshu['beta']='--'
            elif beta>=zhibiao_strategy['beta'].quantile(0.50) and  beta<zhibiao_strategy['beta'].quantile(0.75):
              zhibiao_miaoshu['beta']='--'
            elif beta>=zhibiao_strategy['beta'].quantile(0.75) :
               zhibiao_miaoshu['beta']='--'
               
            #上下行比率
            if updown_ratio<zhibiao_strategy['上下行比率'].astype(float).quantile(0.50):
               zhibiao_miaoshu['上下行比率']='一般'
            elif  updown_ratio>=zhibiao_strategy['上下行比率'].astype(float).quantile(0.50) and  updown_ratio<zhibiao_strategy['上下行比率'].astype(float).quantile(0.75) and  updown_ratio>1:
              zhibiao_miaoshu['上下行比率']='良好'
            elif  updown_ratio>=zhibiao_strategy['上下行比率'].astype(float).quantile(0.75) and  updown_ratio>1 :
               zhibiao_miaoshu['上下行比率']='优秀'
            else:
               zhibiao_miaoshu['上下行比率']='一般'  
            
            #跟踪误差
            if div<zhibiao_strategy['跟踪误差'].astype(float).quantile(0.50):
               zhibiao_miaoshu['跟踪误差']='优秀'
            elif  div>=zhibiao_strategy['跟踪误差'].astype(float).quantile(0.50) and div<zhibiao_strategy['跟踪误差'].astype(float).quantile(0.75) :
              zhibiao_miaoshu['跟踪误差']='良好'
            elif  div>=zhibiao_strategy['跟踪误差'].astype(float).quantile(0.75)  :
               zhibiao_miaoshu['跟踪误差']='一般'
         
            #残差风险
            if r2_risk<zhibiao_strategy['残差风险'].astype(float).quantile(0.50):
               zhibiao_miaoshu['残差风险']='优秀'
            elif  r2_risk>=zhibiao_strategy['残差风险'].astype(float).quantile(0.50) and  r2_risk<zhibiao_strategy['残差风险'].astype(float).quantile(0.75) :
              zhibiao_miaoshu['残差风险']='良好'
            elif  r2_risk>=zhibiao_strategy['残差风险'].astype(float).quantile(0.75)  :
               zhibiao_miaoshu['残差风险']='一般'
          
            #牛市贝塔
            if beta_niu<zhibiao_strategy['牛市贝塔'].quantile(0.50):
               zhibiao_miaoshu['牛市beta']='--'
            elif  beta_niu>=zhibiao_strategy['牛市贝塔'].quantile(0.50) and  beta_niu<zhibiao_strategy['牛市贝塔'].quantile(0.75) and  beta_niu>1:
              zhibiao_miaoshu['牛市beta']='--'
            elif  beta_niu>=zhibiao_strategy['beta'].quantile(0.75) and beta_niu>1 :
               zhibiao_miaoshu['牛市beta']='--'
            else:
               zhibiao_miaoshu['牛市beta']='--' 
            
              #熊市贝塔
            if beta_xiong<zhibiao_strategy['熊市贝塔'].quantile(0.50):
               zhibiao_miaoshu['熊市beta']='--'
            elif  beta_xiong>=zhibiao_strategy['熊市贝塔'].quantile(0.50) and  beta_xiong<zhibiao_strategy['熊市贝塔'].quantile(0.75) :
              zhibiao_miaoshu['熊市beta']='--'
            elif  beta_xiong>=zhibiao_strategy['熊市贝塔'].quantile(0.75):
               zhibiao_miaoshu['熊市beta']='--'
           
            #平均绝对偏差
            if mean_div<zhibiao_strategy['平均绝对偏差'].astype(float).quantile(0.50):
               zhibiao_miaoshu['平均绝对偏差']='优秀'
            elif   mean_div>=zhibiao_strategy['平均绝对偏差'].astype(float).quantile(0.50) and   mean_div<zhibiao_strategy['平均绝对偏差'].astype(float).quantile(0.75) :
              zhibiao_miaoshu['平均绝对偏差']='良好'
            elif   mean_div>=zhibiao_strategy['平均绝对偏差'].astype(float).quantile(0.75) :
               zhibiao_miaoshu['平均绝对偏差']='一般'
            
            #下行风险
            if downrisk<zhibiao_strategy['下行风险'].quantile(0.50):
               zhibiao_miaoshu['下行风险']='优秀'
            elif downrisk>=zhibiao_strategy['下行风险'].quantile(0.50) and  downrisk<zhibiao_strategy['下行风险'].quantile(0.75):
              zhibiao_miaoshu['下行风险']='良好'
            elif downrisk>=zhibiao_strategy['下行风险'].quantile(0.75) :
              zhibiao_miaoshu['下行风险']='一般'
              
                #最大回撤
            if maxdrawdown<zhibiao_strategy['最大回撤'].quantile(0.50):
               zhibiao_miaoshu['最大回撤']='优秀'
            elif maxdrawdown>=zhibiao_strategy['最大回撤'].quantile(0.50) and  maxdrawdown<zhibiao_strategy['最大回撤'].quantile(0.75):
               zhibiao_miaoshu['最大回撤']='良好'
            elif maxdrawdown>=zhibiao_strategy['最大回撤'].quantile(0.75):
               zhibiao_miaoshu['最大回撤']='一般'
       
            #平均回撤
            if meandrawdown<zhibiao_strategy['平均回撤'].astype(float).quantile(0.50):
               zhibiao_miaoshu['平均回撤']='优秀'
            elif meandrawdown>=zhibiao_strategy['平均回撤'].quantile(0.50) and  meandrawdown<zhibiao_strategy['平均回撤'].quantile(0.75):
               zhibiao_miaoshu['平均回撤']='良好'
            elif meandrawdown>=zhibiao_strategy['平均回撤'].quantile(0.75):
               zhibiao_miaoshu['平均回撤']='一般'
                #最大单一回撤  
            if signal_maxdrawdown<zhibiao_strategy['最大单一回撤'].quantile(0.50):
              zhibiao_miaoshu['最大单一回撤']='优秀'
            elif signal_maxdrawdown>=zhibiao_strategy['最大单一回撤'].quantile(0.50) and  signal_maxdrawdown<zhibiao_strategy['最大单一回撤'].quantile(0.75):
              zhibiao_miaoshu['最大单一回撤']='良好'
            elif signal_maxdrawdown>=zhibiao_strategy['最大单一回撤'].quantile(0.75) :
              zhibiao_miaoshu['最大单一回撤']='一般'
      
                #Var
            if VARRatio95<zhibiao_strategy['Var95'].quantile(0.50):
               zhibiao_miaoshu['VaR(95%/10天)']='一般'
            elif VARRatio95>=zhibiao_strategy['Var95'].quantile(0.50) and  VARRatio95<zhibiao_strategy['Var95'].quantile(0.75):
               zhibiao_miaoshu['VaR(95%/10天)']='良好'
            elif VARRatio95>=zhibiao_strategy['Var95'].quantile(0.75) :
              zhibiao_miaoshu['VaR(95%/10天)']='优秀'
       
            
                 #CVar
            if CVar95<zhibiao_strategy['Cvar95'].quantile(0.50):
               zhibiao_miaoshu['CVaR(95%/10天)']='一般'
            elif CVar95>=zhibiao_strategy['Cvar95'].quantile(0.50) and  CVar95<zhibiao_strategy['Cvar95'].quantile(0.75):
              zhibiao_miaoshu['CVaR(95%/10天)']='良好'
            elif CVar95>=zhibiao_strategy['Cvar95'].quantile(0.75):
              zhibiao_miaoshu['CVaR(95%/10天)']='优秀'
                
                 #Var
            if VARRatio99<zhibiao_strategy['Var99'].quantile(0.50):
               zhibiao_miaoshu['VaR(99%)']='一般'
            elif VARRatio99>=zhibiao_strategy['Var99'].quantile(0.50) and  VARRatio99<zhibiao_strategy['Var99'].quantile(0.75):
               zhibiao_miaoshu['VaR(99%)']='良好'
            elif VARRatio99>=zhibiao_strategy['Var99'].quantile(0.75) :
              zhibiao_miaoshu['VaR(99%)']='优秀'
       
            
                 #CVar
            if CVar99<zhibiao_strategy['Cvar99'].quantile(0.50):
               zhibiao_miaoshu['CVaR(99%)']='一般'
            elif CVar99>=zhibiao_strategy['Cvar99'].quantile(0.50) and  CVar99<zhibiao_strategy['Cvar99'].quantile(0.75):
              zhibiao_miaoshu['CVaR(99%)']='良好'
            elif CVar99>=zhibiao_strategy['Cvar99'].quantile(0.75):
              zhibiao_miaoshu['CVaR(99%)']='优秀'
            
            
            #超额收益alpha
            
            if alpha<zhibiao_strategy['超额收益'].quantile(0.50):
               zhibiao_miaoshu['超额收益']='一般'
            elif alpha>=zhibiao_strategy['超额收益'].quantile(0.50) and  alpha<zhibiao_strategy['超额收益'].quantile(0.75) and alpha>0:
              zhibiao_miaoshu['超额收益']='良好'
            elif alpha>=zhibiao_strategy['超额收益'].quantile(0.75) and alpha>0:
              zhibiao_miaoshu['超额收益']='优秀'
            else:
               zhibiao_miaoshu['超额收益']='一般'
            
            #夏普比率
            if Sharp<zhibiao_strategy['夏普比率'].quantile(0.50):
               zhibiao_miaoshu['夏普比率']='一般'
            elif Sharp>=zhibiao_strategy['夏普比率'].quantile(0.50) and  Sharp<zhibiao_strategy['夏普比率'].quantile(0.75) and  Sharp>0:
               zhibiao_miaoshu['夏普比率']='良好'
            elif Sharp>=zhibiao_strategy['夏普比率'].quantile(0.75)  and Sharp>0:
               zhibiao_miaoshu['夏普比率']='优秀'
            else:
               zhibiao_miaoshu['夏普比率']='一般'
            
            #M平方
            if M2<zhibiao_strategy['M平方'].quantile(0.50):
               zhibiao_miaoshu['M平方']='一般'
            elif M2>=zhibiao_strategy['M平方'].quantile(0.50) and  M2<zhibiao_strategy['M平方'].quantile(0.75):
               zhibiao_miaoshu['M平方']='良好'
            elif M2>=zhibiao_strategy['M平方'].quantile(0.75) : 
               zhibiao_miaoshu['M平方']='优秀'
            
            #calmar
            if Calmar<zhibiao_strategy['卡玛比'].quantile(0.50):
               zhibiao_miaoshu['卡玛比率']='一般'
            elif Calmar>=zhibiao_strategy['卡玛比'].quantile(0.50) and  Calmar<zhibiao_strategy['卡玛比'].quantile(0.75):
               zhibiao_miaoshu['卡玛比率']='良好'
            elif Calmar>=zhibiao_strategy['卡玛比'].quantile(0.75): 
              zhibiao_miaoshu['卡玛比率']='优秀'
        
        #斯图泽指数    
       
            if Stutzer<zhibiao_strategy['斯图泽指数'].astype(float).quantile(0.50):
               zhibiao_miaoshu['斯图泽指数']='一般'
            elif Stutzer>=zhibiao_strategy['斯图泽指数'].astype(float).quantile(0.50) and   Stutzer<zhibiao_strategy['斯图泽指数'].astype(float).quantile(0.75) and Stutzer>0:
               zhibiao_miaoshu['斯图泽指数']='良好'
            elif Stutzer>=zhibiao_strategy['斯图泽指数'].astype(float).quantile(0.75) and Stutzer>0: 
              zhibiao_miaoshu['斯图泽指数']='优秀'
            else:
              zhibiao_miaoshu['斯图泽指数']='一般'
            
            #信息比率
            if inforatio<zhibiao_strategy['信息比率'].astype(float).quantile(0.50):
               zhibiao_miaoshu['信息比率']='一般'
            elif inforatio>=zhibiao_strategy['信息比率'].astype(float).quantile(0.50)  and inforatio<zhibiao_strategy['信息比率'].astype(float).quantile(0.75) and  inforatio>0:
               zhibiao_miaoshu['信息比率']='良好'
            elif inforatio>=zhibiao_strategy['信息比率'].astype(float).quantile(0.75)  and  inforatio>0: 
               zhibiao_miaoshu['信息比率']='优秀'
            else:
               zhibiao_miaoshu['信息比率']='一般'
           
            #sterling
            if Sterling<zhibiao_strategy['Sturling指数'].astype(float).quantile(0.50):
              zhibiao_miaoshu['Sterling比率']='一般'
            elif Sterling>=zhibiao_strategy['Sturling指数'].astype(float).quantile(0.50) and Sterling<zhibiao_strategy['Sturling指数'].astype(float).quantile(0.75) and  Sterling>0:
              zhibiao_miaoshu['Sterling比率']='良好'
            elif Sterling>=zhibiao_strategy['Sturling指数'].astype(float).quantile(0.75) and  Sterling>0: 
              zhibiao_miaoshu['Sterling比率']='优秀'
            else:
               zhibiao_miaoshu['Sterling比率']='一般'
           
            #omega指数
            if omega<zhibiao_strategy['Omega比率'].astype(float).quantile(0.50):
              zhibiao_miaoshu['Omega比率']='一般'
            elif omega>=zhibiao_strategy['Omega比率'].astype(float).quantile(0.50) and  omega<zhibiao_strategy['Omega比率'].astype(float).quantile(0.75) :
               zhibiao_miaoshu['Omega比率']='良好'
            elif omega>=zhibiao_strategy['Omega比率'].astype(float).quantile(0.75) : 
               zhibiao_miaoshu['Omega比率']='优秀'
       
            #burke比率
            if burke<zhibiao_strategy['Burke比率'].astype(float).quantile(0.50):
              zhibiao_miaoshu['Burke比率']='一般'
            elif burke>=zhibiao_strategy['Burke比率'].astype(float).quantile(0.50) and  burke<zhibiao_strategy['Burke比率'].astype(float).quantile(0.75) and burke>0:
               zhibiao_miaoshu['Burke比率']='良好'
            elif burke>=zhibiao_strategy['Burke比率'].astype(float).quantile(0.75) and burke>0: 
              zhibiao_miaoshu['Burke比率']='优秀'
            else:
              zhibiao_miaoshu['Burke比率']='一般'
            
            #索提诺比率
            if Sortino<zhibiao_strategy['索提诺比率'].astype(float).quantile(0.50):
               zhibiao_miaoshu['索提诺比率']='一般'
            elif Sortino>=zhibiao_strategy['索提诺比率'].astype(float).quantile(0.50) and  Sortino<zhibiao_strategy['索提诺比率'].astype(float).quantile(0.75) and Sortino>0:
              zhibiao_miaoshu['索提诺比率']='良好'
            elif Sortino>=zhibiao_strategy['索提诺比率'].astype(float).quantile(0.75) and Sortino>0: 
               zhibiao_miaoshu['索提诺比率']='优秀'
            else:
               zhibiao_miaoshu['索提诺比率']='一般'
            treynor
              #索提诺比率
            if treynor<zhibiao_strategy['特雷诺比率'].astype(float).quantile(0.50):
               zhibiao_miaoshu['特雷诺比率']='一般'
            elif treynor>=zhibiao_strategy['特雷诺比率'].astype(float).quantile(0.50) and  treynor<zhibiao_strategy['特雷诺比率'].astype(float).quantile(0.75):
              zhibiao_miaoshu['特雷诺比率']='良好'
            elif treynor>=zhibiao_strategy['特雷诺比率'].astype(float).quantile(0.75) : 
               zhibiao_miaoshu['特雷诺比率']='优秀'
        
            
            #hm择股
            if HM_alpha<zhibiao_strategy['HM择股能力'].astype(float).quantile(0.50):
               zhibiao_miaoshu['HM择股能力']='一般'
            elif HM_alpha>=zhibiao_strategy['HM择股能力'].astype(float).quantile(0.50) and  HM_alpha<zhibiao_strategy['HM择股能力'].astype(float).quantile(0.75) and HM_alpha>0:
              zhibiao_miaoshu['HM择股能力']='良好'
            elif HM_alpha>=zhibiao_strategy['HM择股能力'].astype(float).quantile(0.75) and HM_alpha>0: 
               zhibiao_miaoshu['HM择股能力']='优秀'
            else:
               zhibiao_miaoshu['HM择股能力']='一般'
            #tm择股
            if TM_alpha<zhibiao_strategy['TM择股能力'].astype(float).quantile(0.50):
               zhibiao_miaoshu['TM择股能力']='一般'
            elif TM_alpha>=zhibiao_strategy['TM择股能力'].astype(float).quantile(0.50) and  TM_alpha<zhibiao_strategy['TM择股能力'].astype(float).quantile(0.75) and TM_alpha>0:
              zhibiao_miaoshu['TM择股能力']='良好'
            elif TM_alpha>=zhibiao_strategy['TM择股能力'].astype(float).quantile(0.75) and TM_alpha>0: 
               zhibiao_miaoshu['TM择股能力']='优秀'
            else:
               zhibiao_miaoshu['TM择股能力']='一般'
            
              #CL择股
            if CL_alpha<zhibiao_strategy['CL择股能力'].astype(float).quantile(0.50):
               zhibiao_miaoshu['CL择股能力']='一般'
            elif CL_alpha>=zhibiao_strategy['CL择股能力'].astype(float).quantile(0.50) and  CL_alpha<zhibiao_strategy['CL择股能力'].astype(float).quantile(0.75) and CL_alpha>0:
              zhibiao_miaoshu['CL择股能力']='良好'
            elif CL_alpha>=zhibiao_strategy['CL择股能力'].astype(float).quantile(0.75) and CL_alpha>0: 
               zhibiao_miaoshu['CL择股能力']='优秀'
            else:
               zhibiao_miaoshu['CL择股能力']='一般'
            #贝塔择时比率
            if beta_ratio<zhibiao_strategy['beta择时能力'].astype(float).quantile(0.50):
               zhibiao_miaoshu['beta择时比率']='一般'
            elif beta_ratio>=zhibiao_strategy['beta择时能力'].astype(float).quantile(0.50) and beta_ratio<zhibiao_strategy['beta择时能力'].astype(float).quantile(0.75) and beta_ratio>0:
              zhibiao_miaoshu['beta择时比率']='良好'
            elif beta_ratio>=zhibiao_strategy['beta择时能力'].astype(float).quantile(0.75) and beta_ratio>0: 
               zhibiao_miaoshu['beta择时比率']='优秀'
            else:
               zhibiao_miaoshu['beta择时比率']='一般'
            #CL择时能力
            if CL_beta<zhibiao_strategy['CL择时能力'].astype(float).quantile(0.50):
               zhibiao_miaoshu['CL择时能力']='一般'
            elif  CL_beta>=zhibiao_strategy['CL择时能力'].astype(float).quantile(0.50) and   CL_beta<zhibiao_strategy['CL择时能力'].astype(float).quantile(0.75) and  CL_beta>0:
              zhibiao_miaoshu['CL择时能力']='良好'
            elif  CL_beta>=zhibiao_strategy['CL择时能力'].astype(float).quantile(0.75) and  CL_beta>0: 
               zhibiao_miaoshu['CL择时能力']='优秀'
            else:
               zhibiao_miaoshu['CL择时能力']='一般'
            
             #HM择时能力
            if HM_beta<zhibiao_strategy['HM择时能力'].astype(float).quantile(0.50):
               zhibiao_miaoshu['HM择时能力']='一般'
            elif  HM_beta>=zhibiao_strategy['HM择时能力'].astype(float).quantile(0.50) and   HM_beta<zhibiao_strategy['HM择时能力'].astype(float).quantile(0.75) and HM_beta>0:
              zhibiao_miaoshu['HM择时能力']='良好'
            elif  HM_beta>=zhibiao_strategy['HM择时能力'].astype(float).quantile(0.75) and HM_beta>0: 
               zhibiao_miaoshu['HM择时能力']='优秀'
            else:
               zhibiao_miaoshu['HM择时能力']='一般'
            
              #HM择时能力
            if TM_beta<zhibiao_strategy['TM择时能力'].astype(float).quantile(0.50):
               zhibiao_miaoshu['TM择时能力']='一般'
            elif  TM_beta>=zhibiao_strategy['TM择时能力'].astype(float).quantile(0.50) and  TM_beta<zhibiao_strategy['TM择时能力'].astype(float).quantile(0.75) and TM_beta>0:
              zhibiao_miaoshu['TM择时能力']='良好'
            elif  TM_beta>=zhibiao_strategy['TM择时能力'].astype(float).quantile(0.75) and  TM_beta>0: 
               zhibiao_miaoshu['TM择时能力']='优秀'
            else:
               zhibiao_miaoshu['TM择时能力']='一般'
            return zhibiao_miaoshu
def det_alpha(
        annyet,
        inforatio,
        sortino,
        omega,
        Calmar,
        M2,
        beta,
        mean_drawdown,
        annvol,
        downrisk,
        Cvar95,
        saomiao,
        zhibiao_strategy):
    #各项指标单项评分
     #胜率评分
#    if  win_ratio>=0.7:
#         score['胜率']=100     
#    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
#         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
#    elif win_ratio<=0.31:
#         score['胜率']=0
    score=pd.Series()
#    score['年化收益']=((pd.Series(zhibiao_strategy['年化收益']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['年化收益']=((pd.Series(zhibiao_strategy['return_annual']).astype(float)-annyet)<0).sum()/len(zhibiao_strategy)*100
#    score['超额收益(famma)']=((pd.Series(zhibiao_strategy['alpha']).astype(float)-alpha_famma)<0).sum()/len(zhibiao_strategy)*100
    score['信息比率']=((pd.Series(zhibiao_strategy['information_ratio']).astype(float)-inforatio)<0).sum()/len(zhibiao_strategy)*100
    score['索提诺比率']=((pd.Series(zhibiao_strategy['sortino']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
    score['Omega比率']=((pd.Series(zhibiao_strategy['omega']).astype(float)-omega)<0).sum()/len(zhibiao_strategy)*100
    score['卡玛比']=((pd.Series(zhibiao_strategy['calmar']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M2']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
    score['年化波动率']=((pd.Series(zhibiao_strategy['return_annual']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
    score['beta']=((pd.Series(zhibiao_strategy['beta']).astype(float)-beta)>0).sum()/len(zhibiao_strategy)*100
    score['下行风险']=((pd.Series(zhibiao_strategy['downrisk']).astype(float)-downrisk)>0).sum()/len(zhibiao_strategy)*100
    score['平均回撤']=((pd.Series(zhibiao_strategy['data_mean_drawdown']).astype(float)-mean_drawdown)>0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['data_CVaR95']).astype(float)-Cvar95)>0).sum()/len(zhibiao_strategy)*100
#    score['CL择股能力']=((pd.Series(zhibiao_strategy['CL择股能力']).astype(float)- CL_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['HM择股能力']=((pd.Series(zhibiao_strategy['HM择股能力']).astype(float)- HM_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['TM择股能力']=((pd.Series(zhibiao_strategy['TM择股能力']).astype(float)- TM_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['CL择时能力']=((pd.Series(zhibiao_strategy['CL择时能力']).astype(float)- CL_beta_ratio)<0).sum()/len(zhibiao_strategy)*100
#    score['HM择时能力']=((pd.Series(zhibiao_strategy['HM择时能力']).astype(float)- HM_beta2)<0).sum()/len(zhibiao_strategy)*100
#    score['TM择时能力']=((pd.Series(zhibiao_strategy['TM择时能力']).astype(float)- TM_beta2)<0).sum()/len(zhibiao_strategy)*100
#    score['贝塔择时能力']=((pd.Series(zhibiao_strategy['beta择时能力']).astype(float)- beta_ratio)<0).sum()/len(zhibiao_strategy)*100
    score['业绩一致性']=((pd.Series(zhibiao_strategy['saomiao']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['saomiao'].dropna().size)*100  
    score_radar=pd.Series()
    score_radar['收益能力']=   score['年化收益']
#    score_radar['择时能力']= (score['CL择时能力']+score['HM择时能力']+score['TM择时能力']+ score['贝塔择时能力'])/4
#    score_radar['择股能力']=(score['CL择股能力']+ score['HM择股能力']+ score['TM择股能力'])/3
    score_radar['风控能力']= (score['年化波动率']+score['下行风险']+score['平均回撤']+ score['Cvar95']+score['beta'])/5
    score_radar['风险调整绩效']= (score['索提诺比率']+ score['Omega比率']+score['卡玛比']+ score['M平方']+score['信息比率'])/5
    score_radar['业绩持续性']=  score['业绩一致性']
#    score_radar['行情判断能力']= score['胜率'] 
#    score_scar=pd.Series()
#    score_star=pd.Series()
    score_radar['综合得分']= score_radar['收益能力']*0.30+\
    score_radar['风控能力']*0.25+ score_radar['风险调整绩效']*0.25 +score_radar['业绩持续性']*0.20#+#score_radar['持续性']*0.1
    #星级评定
    star_level=[]
    for i in range(score_radar.size):
        if score_radar[i]>=90:  
                scar_level=5
        elif score_radar[i]<90 and score_radar[i]>=80:
                 scar_level=4.5
        elif score_radar[i]<80 and score_radar[i]>=70:
                 scar_level=4
        elif score_radar[i]<70 and score_radar[i]>=60:
                scar_level=3.5
        elif score_radar[i]>=50 and score_radar[i]<60:
                scar_level=3
        elif score_radar[i]>=40 and score_radar[i]<50:  
                scar_level=2.5
        else:
                scar_level=2          
        star_level.append(scar_level)           
    star_table=pd.Series(star_level,index=['收益能力','风控能力','风险调整绩效','业绩持续性','综合能力'])  
    return score,score_radar, star_table        
def alpha_judge(navs,p,con):   
#    zhibiao_alpha=pd.read_sql('select * from t_strategy_relative',con,index_col='index')     
    sql = "select  *  from t_fund_index2 where  mjfs = '私募' and  strategy = '相对价值' and days_during > 50"
    raw_fof=pd.read_sql(sql,con,index_col='id')  
    raw_fof['qingxi']=(raw_fof['count_nav'].astype(int))/(raw_fof['days_during'].astype(int))> 0.1
    zhibiao_alpha= raw_fof[raw_fof['qingxi']]
    data_indicatrix =alpha_calc_data(navs,p,con)
#    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    score,score_radar,score_table=det_alpha(data_indicatrix['data_annRets'][0],data_indicatrix['data_inforatio'][0],data_indicatrix['data_Sortino'][0],\
         data_indicatrix['omega'][0],data_indicatrix['data_Calmar'][0],data_indicatrix['M2'][0], data_indicatrix['data_beta'][0],data_indicatrix['data_mean_drawdown'][0],\
         data_indicatrix['data_annVol'][0],data_indicatrix['downrisk'][0], data_indicatrix['data_cvar95'][0], data_indicatrix['data_saomiao'][0],zhibiao_alpha)
    return score,score_radar,score_table
    
def turnover_performance(all_stock,everday_marketvalues):
    turnover_holding = pd.concat(list(map(lambda x: x.pivot_table(values='持股数', index=x.index, columns='代码'), (pd.DataFrame(
        all_stock.ix[i, ['代码', '持股数']]) for i in set(all_stock.index)))))
    turnover_holding.sort_index(inplace=True)
    turnover_diff = turnover_holding.fillna(0).diff().fillna(0)
    turnover_close = pd.concat(list(map(lambda x: x.pivot_table(values='收盘价', index=x.index, columns='代码'), (pd.DataFrame(
        all_stock.ix[i, ['代码', '收盘价']]) for i in set(all_stock.index)))))
    turnover_close.sort_index(inplace=True)
    turnover_ratio = (
        turnover_diff.where(
            turnover_diff >= 0, -1 * turnover_diff) * turnover_close / 2).sum(axis=1) / everday_marketvalues  # !!!.shift(1)
    turnover_ratio.name = '换手率'
    #turnover_ratio = pd.concat([turnover_ratio,nav['净值']],axis=1).fillna(0).drop('净值',axis=1)
    return turnover_ratio
    
def interpolate_wrap(x):
    x_dropped = x.dropna()
    if len(x_dropped) < 3:
        return pd.Series()
    else:
        start = x_dropped.index[0]
        end = x_dropped.index[-1]
        x2 = pd.to_numeric(x[start:end])
        return x2.interpolate()
        
def maxdrawdown_and_day(navs):
    navs = pd.DataFrame(navs)
    l = []
    endtimelist=[]
    starttimelist=[]
    for i in range(navs.columns.size):
        nav = navs.ix[:, i].dropna()
        if len(nav) == 0:
            l.append(np.nan)
            starttime=endtime=np.NAN
        else:
            endtime = np.argmax((np.maximum.accumulate(nav) - nav)/np.maximum.accumulate(nav))
            starttime = np.argmax(nav[:endtime])
            high = nav[starttime]
            low = nav[endtime]
            l.append((low - high) / high * -1)
            endtimelist.append(endtime)
            starttimelist.append(starttime)
    if i<=1:
        return l,endtime,starttime
    else:
        return l,endtimelist,starttimelist

        
def Heatmap_multi_nohtml(rets,strategy,hs300_all,navs_and_date_raw_all):
    """
    相关性矩阵产品和策略对比版,同Heatmap_multi一样的，只是不生成html字符串格式
    """
    # 从mysql中读取数据
    if '型' not in strategies_dict[strategy[0]]:
        clzs = get_data_from_mysql('select index_date, \
                                    all_market as 全市场策略, \
                                    macro as 宏观对冲策略, \
                                    relative as 相对价值策略, \
                                    cta as 管理期货策略, \
                                    stocks as 股票多头策略, \
                                    bonds as 固定收益策略  from t_strategy_index', 'index_date')
    else:
        clzs = get_data_from_mysql('select index_date, \
                                    commingled as 混合型, \
                                    bonds as 债券型, \
                                    money as 货币型, \
                                    stocks as 股票型 \
                                    from t_public_strategy_index', 'index_date')
    clzs.index = pd.to_datetime(clzs.index)

    cl = clzs[list(set([strategies_dict[i] for i in list(set(strategy))]))]
    
    if strategy[0] == '股票策略' or strategy[0] == '股票型' or strategy[0] == '股票指数':
        # 从mysql中读取数据
        table = get_data_from_mysql('select index_date, \
                                sz50 as 上证50指数, hs300 as 沪深300指数, zz500 as 中证500指数, zz800 as 中证800指数 \
                                from t_wind_index_market', 'index_date')
        table.index = pd.to_datetime(table.index)
        cl = interpolate2(cl, hs300_all, '日', strategy=None)[0]
        data = pd.concat([navs_and_date_raw_all, cl, table], axis=1, join='outer').dropna()
    elif strategy[0] == '管理期货':
        # 从mysql中读取数据
        table = get_data_from_mysql('SELECT index_date,commodity as 南华商品指数,industry as 南华工业品指数,agriculture as 南华农产品指数, \
                                metal as 南华金属指数,energy_chemical as 南华能化指数, \
                                precious_metal as 南华贵金属指数 FROM south_china_indexes', 'index_date')
        table.index = pd.to_datetime(table.index)
        cl = interpolate2(cl, hs300_all, '日', strategy=None)[0]
        data = pd.concat([navs_and_date_raw_all, cl, table], axis=1, join='outer').dropna()        
    else:
            # 从mysql中读取数据
        table = get_data_from_mysql('select index_date, \
                                hs300 as 沪深300指数, \
                                csi_total_debt as 中证全债指数, \
                                south_china_commodity as 南华商品指数 from t_wind_index_market', 'index_date')
        table.index = pd.to_datetime(table.index)
        cl = interpolate2(cl, hs300_all, '日', strategy=None)[0]
        sc = table[list(set([market_dict[i] for i in list(set(strategy))]))]
        data = pd.concat([navs_and_date_raw_all, cl, sc], axis=1, join='outer').dropna()
    l = []
    for i in range(data.columns.size):
        for j in range(data.columns.size):
            l.append([data.columns.size-1-i, data.columns.size-1-j, round(data.ix[:, [i, j]].dropna().corr(
                method='spearman').fillna(0).values[0, 1],3)])
    cl = str(data.columns.tolist())
    l.reverse()
    return l,cl
    
def InfoRatio2(rets, index_ret, p):

    rets = pd.DataFrame(rets)
    index_ret = pd.DataFrame(index_ret)
    retDF = pd.concat([rets,index_ret],axis=1).dropna()
    excessRet =np.array(retDF.iloc[:,0])-np.array(retDF.iloc[:,1])
    excessRetStd = excessRet.std()
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    result = excessRet.mean() * np.sqrt(period[p[0]]) / excessRetStd
    if result == np.inf:
        result = 9999
    elif result == -np.inf:
        result = -9999
    return result

def nav_completion(navs,szzs,p):
        navs=pd.DataFrame(navs)
        navs_new=pd.DataFrame()
        for i in range(navs.columns.size):
            nav=navs.iloc[:,i].dropna()
            if p[0]=='月':
                szzs_date=szzs.groupby(by=pd.TimeGrouper(freq='BM')).last()
                szzs_date=szzs_date[nav.index[0]:nav.index[-1]]
            else:
               szzs_date=szzs[nav.index[0]:nav.index[-1]] 
            demo=pd.concat([nav,szzs_date],axis=1)
            for j in range(szzs_date.index.size):
                if np.isnan(demo.loc[szzs_date.index[j],demo.columns[0]]):
                   diff_after=nav[nav.index>szzs_date.index[j]].index[0]-szzs_date.index[j]
                   diff_before=szzs_date.index[j]-nav[nav.index<szzs_date.index[j]].index[-1]
                   if diff_after>= diff_before:
                       demo.loc[szzs_date.index[j],demo.columns[0]]=pd.DataFrame(nav[nav.index<szzs_date.index[j]]).iloc[-1,0]
                   else:
                       demo.loc[szzs_date.index[j],demo.columns[0]]=pd.DataFrame(nav[nav.index>szzs_date.index[j]]).iloc[0,0] 
                  
            navs_new=pd.concat([navs_new,demo.iloc[:,0]],axis=1)  
        return navs_new
def Heatmap(rets,strategy=[]):
    if len(rets)<4:
        clzs_simu = get_data_from_mysql('select index_date, \
                                        all_market as 全市场策略, \
                                        macro as 宏观对冲策略, \
                                        relative as 相对价值策略, \
                                        cta as 管理期货策略, \
                                        stocks as 股票多头策略, \
                                        bonds as 固定收益策略  from t_strategy_index', 'index_date')
        clzs_public = get_data_from_mysql('select index_date, \
                                        commingled as 混合型, \
                                        bonds as 债券型, \
                                        money as 货币型, \
                                        stocks as 股票型 \
                                        from t_public_strategy_index', 'index_date') 
        clzs=pd.concat([clzs_simu,clzs_public], axis=1, join='outer')
        cl = clzs[list(set([strategies_dict[i] for i in list(set(strategy))]))]
        cl_rets=calculate_profits(cl)
        
        
        table = get_data_from_mysql('select index_date, \
                                    hs300 as 沪深300指数, \
                                    csi_total_debt as 中证全债指数, \
                                    south_china_commodity as 南华商品指数 from t_wind_index_market', 'index_date')
        table.index = pd.to_datetime(table.index)
        sc = table[list(set([market_dict[i] for i in list(set(strategy))]))]
        sc_rets=calculate_profits(sc)
        
        data = pd.concat([ sc_rets, cl_rets,rets], axis=1, join='outer')
    elif len(rets)<6:
        table = get_data_from_mysql('select index_date, \
                                    hs300 as 沪深300指数, \
                                    csi_total_debt as 中证全债指数, \
                                    south_china_commodity as 南华商品指数 from t_wind_index_market', 'index_date')
        table.index = pd.to_datetime(table.index)
        sc = table[list(set([market_dict[i] for i in list(set(strategy))]))]
        sc_rets=calculate_profits(sc)
        
        data = pd.concat([ sc_rets,rets], axis=1, join='outer') 
    else:
        data=rets
    return data.corr(method='spearman')  
def gui1(navs):
      navs=pd.DataFrame(navs)
      nav_1=pd.DataFrame()
      for i in range(navs.columns.size):
        nav=navs.iloc[:,i].dropna()
        nav_new=nav/ nav[0]
        nav_1=pd.concat([nav_1, nav_new],axis=1)
      return nav_1
def UC(navs,market_index):
    l=[]
    navs=pd.DataFrame(navs)
    for i in range(navs.columns.size):
        nav=navs.iloc[:,0].dropna()
        retms=nav.resample('M').last().pct_change().dropna()
        retms=pd.DataFrame(retms)
        market_retms=market_index.resample('M').last().pct_change().dropna()
        market_retms=pd.DataFrame(market_retms)
        upp=market_retms>=0
     
        upp=pd.concat([retms,upp],join='inner', axis=1).T.ix[1:2].T
        lens=upp.sum()[0]
        retms_up=(retms.iloc[:,0]*upp.ix[:,0]).dropna()
        lens=retms_up.sum()
        market_retms_up=(market_retms.iloc[:,0]*upp.ix[:,0]).dropna()
        UCR=((1+retms_up).cumprod()[-1])**(1/lens)-1
        UCRm=((1+market_retms_up).cumprod()[-1])**(1/lens)-1
        uc=UCR/UCRm*100
        l.append(uc)
    return l
#下行捕获率    
def DC(navs,market_index):
    l=[]
    navs=pd.DataFrame(navs)
    for i in range(navs.columns.size):
        nav=navs.iloc[:,i].dropna()
        retms=nav.resample('M').last().pct_change().dropna()
        retms=pd.DataFrame(retms)
        market_retms=market_index.resample('M').last().pct_change().dropna()
        market_retms=pd.DataFrame(market_retms)
        dnn=(market_retms<0)
        dnn=pd.concat([retms,dnn],join='inner', axis=1).T.ix[1:2].T
        lens=dnn.sum()[0]

        retms_dn=(retms.ix[:,0]*dnn.ix[:,0]).dropna()
        market_retms_dn=(market_retms.ix[:,0]*dnn.ix[:,0]).dropna()
        DCR=((1+retms_dn).cumprod()[-1])**(1/lens)-1
        DCRm=((1+market_retms_dn).cumprod()[-1])**(1/lens)-1
        dc=DCR/DCRm*100
        l.append(dc)
    return l
         