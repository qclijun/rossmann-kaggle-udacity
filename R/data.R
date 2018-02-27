train_csv_file = '../input/train.csv'
store_csv_file = '../input/store.csv'
test_csv_file = '../input/test.csv'
store_states_csv_file = '../input/store_states.csv'


# preprocess store
store = fread(store_csv_file)
store_states = fread(store_states_csv_file)
# merge states
store = merge(store, store_states)
rm(store_states)
store[, StoreType:=factor(StoreType, levels = c('a','b','c','d'))]
store[, StoreTypeN:=as.integer(StoreType)]
store[, Assortment:=factor(Assortment, levels = c('a','b','c','d'))]
store[, AssortmentN:=as.integer(Assortment)]
store[, State:=factor(State)]
store[, StateN:=as.integer(State)]
store[is.na(CompetitionDistance), CompetitionDistance:=0]
store[, CompetitionDistance:=log1p(CompetitionDistance)/10]

store[, CompetitionOpenDate:=as.Date(
  paste(CompetitionOpenSinceYear, CompetitionOpenSinceMonth, 15, sep='-'),
  format='%Y-%m-%d'
)]
store[, Promo2StartDate:=as.Date(
  paste(Promo2SinceYear, Promo2SinceWeek, 1, sep='-'),
  format='%Y-%U-%u'
)]
store[, Promo2IntervalN:=as.integer(
  factor(substr(PromoInterval,1,1), levels=c('J','F','M',''))
)]
store[Promo2IntervalN==4, Promo2IntervalN:=0]


# train, test
train = fread(train_csv_file)
test = fread(test_csv_file)

train[, Date:=as.Date(Date, format='%Y-%m-%d')]
test[, Date:=as.Date(Date, format='%Y-%m-%d')]

stores_not_in_test = setdiff(seq(1:1115), unique(test$Store))
stores_in_test = unique(test$Store)
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


all_data = rbindlist(list(train, miss_records, test), fill=TRUE)
all_data[is.na(Open), Open:=1]
#all_data[, Date:=as.Date(Date, format='%Y-%m-%d')]
all_data[, SchoolHoliday:=as.integer(SchoolHoliday)]
setkey(all_data, 'Date','Store')

# base_features
all_data[, Year:=year(Date)]
all_data[, Month:=month(Date)]
all_data[, Day:=mday(Date)]
all_data[, WeekOfYear:=week(Date)]
all_data[, StateHolidayN:=as.integer(factor(StateHoliday, levels=c('a','b','c')))]
all_data[is.na(StateHolidayN), StateHolidayN:=0]



# merge all_data with store
all_data = merge(all_data, store, by='Store')


all_data[, CompeteOpenMonths:=as.integer((Date-CompetitionOpenDate)/30)]
all_data[is.na(CompeteOpenMonths) | CompeteOpenMonths<0, CompeteOpenMonths:=0]
all_data[CompeteOpenMonths>24, CompeteOpenMonths:=24]

all_data[, Promo2OpenWeeks:=as.integer((Date-Promo2StartDate)/7)]
all_data[is.na(Promo2OpenWeeks)|Promo2OpenWeeks<0, Promo2OpenWeeks:=0]
all_data[Promo2OpenWeeks>25, Promo2OpenWeeks:=25]


mask = all_data$Month < all_data$Promo2IntervalN
all_data[mask, `:=`(Latest_Promo2_Start_Year=Year-1,
                    Latest_Promo2_Start_Month=Promo2IntervalN+12-3)]
all_data[!mask, `:=`(Latest_Promo2_Start_Year=Year,
                     Latest_Promo2_Start_Month=as.integer((Month-Promo2IntervalN)/3)*3+Promo2IntervalN)]
all_data[, Latest_Promo2_Month:=as.integer(
  (Date-as.Date(
  paste(Latest_Promo2_Start_Year, Latest_Promo2_Start_Month, 1, sep='-'),
  format='%Y-%m-%d')
  )/30
)]
all_data[Promo2IntervalN==0, Latest_Promo2_Month:=0]


# merge weather data
all_data = merge(all_data, weather_data, by=c('State', 'Date'), all.x=TRUE, all.y=FALSE)

# merge googletrend
all_data = merge(all_data, trend_data[State=='DE', .(Year, WeekOfYear, Trend_Val)], by=c('Year', 'WeekOfYear'),
                 all.x=TRUE, all.y=FALSE)
names(all_data)[names(all_data)=='Trend_Val'] = 'Trend_Val_DE'
all_data = merge(all_data, trend_data[State!='DE'], by=c('State', 'Year', 'WeekOfYear'),
                 all.x=TRUE, all.y=FALSE)
names(all_data)[names(all_data)=='Trend_Val'] = 'Trend_Val_State'

trend_val_de_min = min(all_data[is.na(Id), Trend_Val_DE])
trend_val_de_max = max(all_data[is.na(Id), Trend_Val_DE])
trend_val_state_min = min(all_data[is.na(Id), Trend_Val_State])
trend_val_state_max = max(all_data[is.na(Id), Trend_Val_State])

all_data[, Trend_Val_DE:=(Trend_Val_DE - trend_val_de_min)/(trend_val_de_max - trend_val_de_min)]
all_data[, Trend_Val_State:=(Trend_Val_State - trend_val_state_min)/(trend_val_state_max - trend_val_state_min)]


avg_sales = cal_avg_sales(all_data)
all_data = merge(all_data, avg_sales[, .(Store, Sales_Per_Day, Customers_Per_Day, Sales_Per_Customer)], by='Store')

all_data[, Sales_Log:=log1p(Sales)]
all_data[is.na(CompetitionOpenSinceYear), CompetitionOpenSinceYear:=0]
all_data[is.na(Promo2SinceYear), Promo2SinceYear:=0]

all_data = merge(all_data, fb2, by=c('Date', 'Store'), all.x=TRUE, all.y=FALSE)

# promo_decay feature

add_promo_decay_feature(all_data)


all_data[, TomorrowClosed:=shift(Open, fill=1, type='lead')==0, by='Store']
all_data[, CompetitionDistance_Log:=log1p(CompetitionDistance)/10]

setkey(all_data, 'Date','Store')
features = c('Id',
             'Store', 'DayOfWeek', 'Promo', 'Year', 'Month',
             'Day', 'StateHolidayN', 'SchoolHoliday', 'CompeteOpenMonths', 'Promo2OpenWeeks',
             'Latest_Promo2_Start_Month', 'CompetitionDistance_Log', 'StoreTypeN', 'AssortmentN', 'Promo2IntervalN',
             'CompetitionOpenSinceYear', 'Promo2SinceYear', 'StateN', 'WeekOfYear', 'Max_TemperatureC',
             'Mean_TemperatureC', 'Min_TemperatureC', 'Max_Humidity', 'Mean_Humidity', 'Min_Humidity',
             'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h', 'CloudCover', 'Events', 'Promo_Forward',
             'Promo_Backward', 'StateHoliday_Forward', 'StateHoliday_Backward', 'StateHoliday_Count_Forward', 'StateHoliday_Count_Backward',
             'SchoolHoliday_Forward', 'SchoolHoliday_Backward', 'Trend_Val_DE', 'Trend_Val_State',
             'Sales_Per_Day', 'Customers_Per_Day', 'Sales_Per_Customer', 'PromoDecay', 'TomorrowClosed'
             )



save_dt(all_data, c(features,'Before_Long_Closed', 'After_Long_Closed', 'Sales'), '../output/featured_data4.h5') 
save_dt(all_data, c(features, 'Sales'), '../output/featured_data_filtered.h5',filter_outliers=TRUE) 


