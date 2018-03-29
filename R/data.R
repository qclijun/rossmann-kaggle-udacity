source('function.R')

train_csv_file = '../input/train.csv'
store_csv_file = '../input/store.csv'
test_csv_file = '../input/test.csv'
store_states_csv_file = '../input/store_states.csv'

weather_csv_dir = '../weather/'
googletrend_csv_dir = '../googletrend/'

store_csv = fread(store_csv_file)
train_csv = fread(train_csv_file)
test_csv = fread(test_csv_file)


# preprocess store
store = copy(store_csv)
store_states = fread(store_states_csv_file)
# merge states
store = merge(store, store_states)
rm(store_states)
store[, StoreType:=factor(StoreType, levels = c('a','b','c','d'))]
store[, StoreTypeN:=as.integer(StoreType)]
store[, Assortment:=factor(Assortment, levels = c('a','b','c'))]
store[, AssortmentN:=as.integer(Assortment)]
store[, State:=factor(State)]
store[, StateN:=as.integer(State)]
store[is.na(CompetitionDistance), CompetitionDistance:=0]
store[, CompetitionDistance:=log1p(CompetitionDistance)/10]

# CompetitionOpen
store[is.na(CompetitionOpenSinceYear), CompetitionOpenSinceYear:=2010]
store[is.na(CompetitionOpenSinceMonth), CompetitionOpenSinceMonth:=8]

store[, CompetitionOpenDate:=as.Date(
  paste(CompetitionOpenSinceYear, CompetitionOpenSinceMonth, 15, sep='-'),
  format='%Y-%m-%d'
)]

# Promo2Start

store[is.na(Promo2SinceYear), Promo2SinceYear:=2012]
store[is.na(Promo2SinceWeek), Promo2SinceWeek:=25]

store[, Promo2StartDate:=as.Date(
  paste(Promo2SinceYear, Promo2SinceWeek, 1, sep='-'),
  format='%Y-%U-%u'
)]
store[, Promo2IntervalN:=as.integer(
  factor(substr(PromoInterval,1,1), levels=c('J','F','M'))
)]
store[is.na(Promo2IntervalN), Promo2IntervalN:=0]


# train, test
train = copy(train_csv)
test = copy(test_csv)

train[, Date:=as.Date(Date, format='%Y-%m-%d')]
test[, Date:=as.Date(Date, format='%Y-%m-%d')]

stores_in_test = unique(test$Store)
stores_not_in_test = setdiff(seq(1:1115), stores_in_test)

train_record_nums = train[, .N, by=Store]
stores_miss_1_record = 988 # miss record 2013-1-1
stores_miss_half_year_records = train_record_nums[N==758]$Store #180 stores miss records 2014-7-1 - 2015-1-1(not include)

miss_date_range = seq(as.Date('2014-7-1'), as.Date('2014-12-31'), 'days')

miss_records = merge.data.frame(
  data.table(Store=stores_miss_half_year_records),
  data.table(DayOfWeek=wday(miss_date_range), Date=miss_date_range,
             Sales=0, Customers=0, Open=0,
             Promo=0, StateHoliday='0', SchoolHoliday=0)
)
miss_records = rbind(miss_records, data.frame(Store=stores_miss_1_record,
                                              DayOfWeek=wday(as.Date('2013-1-1')),
                                              Date=as.Date('2013-1-1'),
                                              Sales=0,
                                              Customers=0,
                                              Open=0,
                                              Promo=0,
                                              StateHoliday='0',
                                              SchoolHoliday=0
                                              ))
miss_records = data.table(miss_records)
# fill missing records
train_cartesian = rbindlist(list(train, miss_records), fill=TRUE)
all_data = rbindlist(list(train_cartesian, test), fill=TRUE)
rm(train_cartesian)


all_data[is.na(Open), Open:=1]
all_data[, SchoolHoliday:=as.integer(SchoolHoliday)]
all_data[, StateHolidayN:=as.integer(factor(StateHoliday, levels=c('a','b','c')))]
all_data[is.na(StateHolidayN), StateHolidayN:=0]


# base_features
all_data[, Year:=year(Date)]
all_data[, Month:=month(Date)]
all_data[, Day:=mday(Date)]
all_data[, WeekOfYear:=week(Date)]
all_data[, DayOfYear:= yday(Date)]



# merge all_data with store
all_data = merge(all_data, store, by='Store')
setkey(all_data, 'Date','Store')

# CompetOpenMonths:[-2, 24]
all_data[, CompeteOpenMonths:=as.integer((Date-CompetitionOpenDate)/30)]
all_data[CompeteOpenMonths<=-2, CompeteOpenMonths:=-2]
all_data[CompeteOpenMonths>24, CompeteOpenMonths:=24]

# Promo2OpenWeeks: [-2, 25]
all_data[, Promo2OpenWeeks:=as.integer((Date-Promo2StartDate)/7)]
all_data[Promo2OpenWeeks<=-2, Promo2OpenWeeks:=-2]
all_data[Promo2OpenWeeks>25, Promo2OpenWeeks:=25]


all_data[, `:=`(Latest_Promo2_Start_Year=Year,
                     Latest_Promo2_Start_Month=as.integer((Month-Promo2IntervalN)/3)*3+Promo2IntervalN)]
mask = all_data$Month < all_data$Promo2IntervalN
all_data[mask, `:=`(Latest_Promo2_Start_Year=Year-1,
                    Latest_Promo2_Start_Month=Promo2IntervalN+12-3)]

# Latest_Promo2_Month:[0, 1, 2, 3], 3 means NA(not participate Promo2)
all_data[, Latest_Promo2_Month:=(Year - Latest_Promo2_Start_Year)*12 + (Month - Latest_Promo2_Start_Month)]
all_data[Promo2IntervalN==0, Latest_Promo2_Month:=3]


open_days = all_data[Open==1&is.na(Id), .N, by=Store]


# merge weather data
weather_data = parse_weather(weather_csv_dir)
all_data = merge(all_data, weather_data, by=c('State', 'Date'), all.x=TRUE, all.y=FALSE)
features_weather = c('Max_TemperatureC',
                     'Mean_TemperatureC', 'Min_TemperatureC', 'Max_Humidity', 'Mean_Humidity', 'Min_Humidity',
                     'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h', 'CloudCover', 'Events')


# merge googletrend
trend_data = parse_googletrend(googletrend_csv_dir)
all_data = merge(all_data, trend_data[State=='DE', .(Year, WeekOfYear, Trend_Val)], by=c('Year', 'WeekOfYear'),
                 all.x=TRUE, all.y=FALSE)
names(all_data)[names(all_data)=='Trend_Val'] = 'Trend_Val_DE'
all_data = merge(all_data, trend_data[State!='DE'], by=c('State', 'Year', 'WeekOfYear'),
                 all.x=TRUE, all.y=FALSE)
names(all_data)[names(all_data)=='Trend_Val'] = 'Trend_Val_State'

features_trend = c('Trend_Val_DE', 'Trend_Val_State')
all_data[, c(features_trend):=lapply(.SD, min_max_scaler), .SDcols=features_trend]

shopavg = cal_shopavg()
all_data = merge(all_data, shopavg, by='Store')
features_shopavg = c("Sales_Per_Day", "Customers_Per_Day", "Sales_Per_Customer", "Sales_Promo", "Sales_Holiday",
                     "Sales_Saturday", "Open_Ratio", "SchoolHoliday_Ratio")

add_fb_features(time_window=7)
features_fb = c('Promo_Backward', 'Promo_Forward', 'SchoolHoliday_Backward', 'SchoolHoliday_Forward',
                'StateHoliday_Backward', 'StateHoliday_Forward')

add_period_count_features(period=7)
features_count_fb = c('Promo_Count_BW', 'Promo_Count_FW', 'SchoolHoliday_Count_BW', 'SchoolHoliday_Count_FW',
                      'StateHoliday_Count_BW', 'StateHoliday_Count_FW')

# promo_decay feature
add_promo_decay_feature()

all_data[, TomorrowClosed:=shift(Open, fill=1, type='lead')==0, by='Store']
all_data[, Is_Dec:=as.integer(Month==12)]
# Long_Closed feature
add_long_closed_features(all_data)

all_data[, Days:=as.integer(Date-as.Date('2013-1-1'))]

all_data[, Log_Customers:=log(Customers)/8.908]

features = c('Id',
             'Store', 'DayOfWeek', 'Promo', 'Year', 'Month',
             'Day', 'WeekOfYear', 'DayOfYear', 'StateHolidayN', 'SchoolHoliday', 'CompeteOpenMonths', 'Promo2OpenWeeks',
             'Latest_Promo2_Start_Month', 'CompetitionDistance', 'StoreTypeN', 'AssortmentN', 'Promo2IntervalN',
             'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'Promo2SinceYear', 'Promo2SinceWeek',
             'StateN','Trend_Val_DE', 'Trend_Val_State',
             'PromoDecay', 'TomorrowClosed',
             'Before_Long_Closed', 'After_Long_Closed',
             'Is_Dec', 'Days'
             )

add_summer_holiday_features()
features_summerholiday = c('Before_SummerHoliday_Start', 'After_SummerHoliday_Start', 'Before_SummerHoliday_End', 'After_SummerHoliday_End')

#features_quar_mean = c('Quar_Mean_Customers', 'Quar_Mean_Sales', 'Quar_Mean_Customers_P', 'Quar_Mean_Sales_P', 
#                       'Quar_Mean_Customers_DW', 'Quar_Mean_Sales_DW')

#all_data = get_mean_sales()
setkey(all_data, 'Date','Store')

save_dt(all_data, c(features, features_weather, features_shopavg, features_fb, features_count_fb, features_summerholiday, 'Log_Customers', 'Sales'), '../output/all_data.h5')



