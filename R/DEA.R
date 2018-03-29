library(data.table)
library(zoo)

library(forecast)
library(ggplot2)

test <- fread("../input/test.csv")
train <- fread("../input/train.csv")
store <- fread("../input/store.csv")

summary(train)
summary(test)
test[is.na(Open)]
test[Store==622,Open]
test[is.na(Open), Open:=1]

test[Store==622,Open]

train[, lapply(.SD, function(x) length(unique(x)))]
test[, lapply(.SD, function(x) length(unique(x)))]

sum(test[, unique(Store)] %in% train[, unique(Store)])
sum(!(train[, unique(Store)] %in% test[, unique(Store)]))

train[, mean(Open)]
test[, mean(Open)]

train[, mean(Promo)]
test[, mean(Promo)]

n_train = nrow(train)
n_test = nrow(test)

train[, .(proba=.N/n_train), by=StateHoliday]
test[, .(proba=.N/n_test), by=StateHoliday]

train[, .(proba=.N/n_train), by=SchoolHoliday][order(SchoolHoliday)]
test[, .(proba=.N/n_test), by=SchoolHoliday][order(SchoolHoliday)]

tmp <- train[, .(ClosedDays=sum(!Open)), by=Store]
tmp[ClosedDays==0]

tmp2 <- test[, .(ClosedDays=sum(!Open)), by=Store]
tmp2[ClosedDays==0]


train[, Date:=as.POSIXct(Date)]
test[, Date:=as.POSIXct(Date)]

store[, Promo2Since:=as.POSIXct(paste(Promo2SinceYear, Promo2SinceWeek, 1, sep=" "), format="%Y %U %u")]
store[, CompetitionOpenSince:=as.POSIXct(
  paste(CompetitionOpenSinceYear, CompetitionOpenSinceMonth, 1, sep=" "), format="%Y %m %d")]

head(store)

train_store <- merge(train, store, by="Store")
test_store <- merge(test, store, by="Store")

setkey(train_store, Store, Date)
setkey(test_store, Store, Date)

date_range <- function(min, max){
  seq(as.POSIXct(min), as.POSIXct(max), by="DSTday")
}

hist(train_store[, mean(Open), by=Store]$V1, 100)

ggplot(train_store[.(103, date_range("2014-1-1", "2015-1-1")), .(Date, Open)], aes(x=Date, y=Open)) +
  geom_point(size=0.1)


# Sales
hist(train$Sales, 100)
# mean_sales per store
mean_sales_customers <- train[Sales!=0, .(mean_sales=mean(Sales),
                                mean_customers=mean(Customers)), by=Store]
hist(mean_sales$mean_sales, 100)

hist(train$Customers, 100)
hist(mean_sales_customers$mean_customers, 100)

ggplot(train[Sales!=0], aes(x=factor(SchoolHoliday),y=Sales)) + 
  geom_jitter(alpha=0.1) + 
  geom_boxplot(color='yellow', outlier.color = NA, fill=NA)

sales_vs_SchoolHoliday <- train[Sales!=0, .(mean(Sales), 
                                            min(Sales), max(Sales)
                                                    ), by=SchoolHoliday]

plot(train[Store==972, Sales], ylab='Sales', xlab='Days', main="Store 972")
plot(train[Store==103, Sales], ylab='Sales', xlab='Days', main="Store 103")
plot(train[Store==708, Sales], ylab='Sales', xlab='Days', main="Store 708")

tmp <- train[, .(ClosedDays=sum(!Open)), by=Store]
tmp[ClosedDays==0]
ggplot(train[Store==85], aes(x=Date, y=Sales, color=factor(DayOfWeek==7),
                             shape=factor(DayOfWeek==7))) +
  geom_point(size=3) + ggtitle("Sales of store 85(True if Sunday)")

ggplot(train[Store==262], aes(x=Date, y=Sales, color=factor(DayOfWeek==7),
                             shape=factor(DayOfWeek==7))) +
  geom_point(size=3) + ggtitle("Sales of store 262(True if Sunday)")

ggplot(train[Sales != 0],
       aes(x = factor(DayOfWeek), y = Sales)) + 
  geom_jitter(alpha = 0.1) + 
  geom_boxplot(color = "yellow", outlier.colour = NA, fill = NA)


# store
summary(store)

store[, .N, by=StoreType]
store[, .N, by=Assortment]
table(data.frame(Assortment=store$Assortment, StoreType=store$StoreType))

hist(store$CompetitionDistance, 100)
store[,CompetitionOpenSince:=as.yearmon(
  paste(CompetitionOpenSinceYear, CompetitionOpenSinceMonth, sep='-'))]

hist(as.yearmon("2015-10")-store$CompetitionOpenSince, 100, 
     main="Years since opening of nearest competition")
summary(store$CompetitionOpenSince)

store[, Promo2Since:=as.POSIXct(paste(Promo2SinceYear, Promo2SinceWeek, 1, sep=" "), format="%Y %U %u")]
hist(as.numeric(as.POSIXct("2015-10-01", format="%Y-%m-%d")-store$Promo2Since), 100,
     main='Days since start of promo2')

table(store$PromoInterval)
store[, .N, by=PromoInterval]

train_store <- merge(train, store, by="Store")

ggplot(train_store[Sales!=0], aes(x=factor(PromoInterval), y=Sales)) + 
  geom_jitter(alpha=0.1) +
  geom_boxplot(color="yellow", outlier.color = NA, fill=NA)

ggplot(train_store[Sales!=0], aes(x=as.Date(Date), y=Sales, color=factor(StoreType))) +
  geom_smooth(size=2)

ggplot(train_store[Sales!=0], aes(x=as.Date(Date), y=Customers, color=factor(StoreType)))+
  geom_smooth(size=2)

ggplot(train_store[Sales!=0], aes(x=as.Date(Date), y=Sales, color=factor(Assortment))) + 
  geom_smooth(size=2)

ggplot(train_store[Sales!=0], aes(x=as.Date(Date), y=Customers, color=factor(Assortment))) + 
  geom_smooth(size=2)

salesByDist <- train_store[Sales!=0 & !is.na(CompetitionDistance), .(CompetitionDistance, Sales)]
salesByDist <- salesByDist[, .(mean_sales=mean(Sales)), by=CompetitionDistance]
ggplot(salesByDist, aes(x=log(CompetitionDistance), y=log(mean_sales))) + 
  geom_point() + geom_smooth()

ggplot(train_store[Sales!=0], aes(x=factor(!is.na(CompetitionOpenSinceYear)), y=Sales)) +
  geom_jitter(alpha=0.1) +
  geom_boxplot(color='yellow', outlier.color = NA, fill=NA) +
  ggtitle("Any competition?")


train_store[, Date:=as.POSIXct(Date, format='%Y-%m-%d')]
train_store[, Year:=year(Date)]
train_store[, Month:=month(Date)]
train_store[, Day:=month(Date)]
train_store[, StoreMean:= mean(Sales), by=Store]
train_store[, MonthlySalesMean:=mean(Sales/StoreMean)*100, by=.(Year, Month)]

SalesTS <- ts(train_store$MonthlySalesMean, start=2013, frequency = 12)
col = rainbow(3)
seasonplot(SalesTS, col=col, year.labels.left = TRUE, pch=19, las=1)

temp <- train
temp[, Date:=as.POSIXct(Date, format="%Y-%m-%d")]
temp[, Year:=year(Date)]
temp[, Month:=month(Date)]
temp[,StoreMean:=mean(Sales), by=Store]
temp<- temp[,.(MonthlySalesMean=mean(Sales/StoreMean)*100), by=.(Year, Month)]

SalesTS <- ts(temp$MonthlySalesMean, start=2013, frequency = 12)
SalesTS
col=rainbow(3)
seasonplot(SalesTS, col=col, year.labels.left = TRUE, pch=19, las=1)
