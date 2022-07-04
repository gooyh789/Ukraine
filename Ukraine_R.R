getwd()
setwd("~/R/Ukraine")
library(readr)
# Data is May 30 version....
#df <- read_csv("warData.csv")
df1 <- read_csv("warData1.csv") # Raw data
#df2 <- read_csv("warData2.csv")
#intersection <- read_csv("intersection.csv")
#simple <- read_csv("total.csv")
simple1 <- read_csv("total1.csv") # Clustering result
#new <- read_csv("new.csv")

library(dplyr)
df1_map <- df1 %>% select(geonameid, city, initial, longitude, latitude, current, occupy_change, contested)

library(Rcpp)
library(tidyverse)
library(sf)
library(mapview)

# Mapping the cities considering the combat duration before March 29
max(df1_map['contested']) # 25

mapview(df1_map, xcol = "longitude", ycol = "latitude", crs = 4269, grid = FALSE, 
        zcol = "contested")

mapview(df1_map, xcol = "longitude", ycol = "latitude", crs = 4269, grid = FALSE, 
        zcol = "current")

mapview(simple1, xcol = "longitude", ycol = "latitude", crs = 4269, grid = FALSE, 
        zcol = "current", layer.name = 'Current Status')

simple1 <- simple1 %>% filter(latitude < 53)
mapview(simple1, xcol = "longitude", ycol = "latitude", crs = 4269, grid = FALSE, 
        zcol = "segment", layer.name = 'Cluster')

mapview(simple1, xcol = "longitude", ycol = "latitude", crs = 4269, grid = FALSE, 
        zcol = "current", layer.name = 'current')



cluster_center <- simple1 %>% filter(segment == 'Cluster 2' | segment == 'Cluster 3')
mapview(cluster_center, xcol = "longitude", ycol = "latitude", crs = 4269, grid = FALSE,
        zcol="segment")

mapview(cluster_center, xcol = "longitude", ycol = "latitude", crs = 4269, grid = FALSE,
        zcol="current")






north_tot <- df1 %>% filter(longitude <= 35.5 & latitude >= 49)
east_tot <- df1 %>% filter(longitude > 35.5 | latitude < 49)
south_tot <- east_tot %>% filter(longitude <= 36.8 & latitude < 48)
east_tot <- setdiff(east_tot,south_tot) %>% filter(latitude < 54)


north_t <- north_tot %>% select(-geonameid, -city, -longitude, -latitude, -feature_code, -initial, -current,
                                -occupy_change, -contested)

ua_oc <- c()
ru_oc <- c()
con_oc <- c()
for (i in 1:ncol(north_t)) {
  ua = 0
  ru = 0
  con = 0
  for (j in 1:nrow(north_t)) {
    if (north_t[j,i] == 'UA') {
      ua = ua + 1
    } else if (north_t[j,i] == 'RU') {
      ru = ru + 1
    } else if (north_t[j,i] == 'CONTESTED') {
      con = con + 1
    }
  }
  ua_oc <- append(ua_oc, ua)
  ru_oc <- append(ru_oc, ru)
  con_oc <- append(con_oc, con)
}
north_occupy <- data.frame(as.Date.character(colnames(north_t), '%Y%m%d'),ua_oc, ru_oc, con_oc)
colnames(north_occupy) <- c('date', 'ua_north', 'ru_north', 'con_north')
east_t <- east_tot %>% select(-geonameid, -city, -longitude, -latitude, -feature_code, -initial, -current,
                              -occupy_change, -contested)

ua_oc <- c()
ru_oc <- c()
con_oc <- c()
for (i in 1:ncol(east_t)) {
  ua = 0
  ru = 0
  con = 0
  for (j in 1:nrow(east_t)) {
    if (east_t[j,i] == 'UA') {
      ua = ua + 1
    } else if (east_t[j,i] == 'RU') {
      ru = ru + 1
    } else if (east_t[j,i] == 'CONTESTED') {
      con = con + 1
    }
  }
  ua_oc <- append(ua_oc, ua)
  ru_oc <- append(ru_oc, ru)
  con_oc <- append(con_oc, con)
}
east_occupy <- data.frame(ua_oc, ru_oc, con_oc)
colnames(east_occupy) <- c('ua_east', 'ru_east', 'con_east')

south_t <- south_tot %>% select(-geonameid, -city, -longitude, -latitude, -feature_code, -initial, -current,
                                -occupy_change, -contested)

ua_oc <- c()
ru_oc <- c()
con_oc <- c()
for (i in 1:ncol(south_t)) {
  ua = 0
  ru = 0
  con = 0
  for (j in 1:nrow(south_t)) {
    if (south_t[j,i] == 'UA') {
      ua = ua + 1
    } else if (south_t[j,i] == 'RU') {
      ru = ru + 1
    } else if (south_t[j,i] == 'CONTESTED') {
      con = con + 1
    }
  }
  ua_oc <- append(ua_oc, ua)
  ru_oc <- append(ru_oc, ru)
  con_oc <- append(con_oc, con)
}
south_occupy <- data.frame(ua_oc, ru_oc, con_oc)
colnames(south_occupy) <- c('ua_south', 'ru_south', 'con_south')

tot_occupy <- cbind(north_occupy, east_occupy, south_occupy)

library(ggplot2)
colors <- c("Center" = "red", "East" = "blue", "South" = "black")
lty <- c("Russian" = "solid", "Contested" = "dotted")
fill <- c("Center" = "red", "East" = "blue", "South" = "black")
ggplot(data=tot_occupy, aes(x=date)) +
  geom_area(aes(y=con_north+ru_north, fill='Center'), size=1, alpha = 0.1) +
  geom_area(aes(y=con_east+ru_east, fill='East'), size=1, alpha = 0.1) +
  geom_area(aes(y=con_south+ru_south, fill='South'), size=1, alpha = 0.1) +
  geom_line(aes(y=ru_east, color = 'East', linetype = 'Russian'), size=1) +
  geom_line(aes(y=ru_north, color = 'Center', linetype = 'Russian'), size=1) +
  geom_line(aes(y=ru_south, color = 'South', linetype = 'Russian'), size=1) +
  geom_line(aes(y=con_east, color = 'East', linetype = 'Contested'), size=1) +
  geom_line(aes(y=con_north, color = 'Center', linetype = 'Contested'), size=1) +
  geom_line(aes(y=con_south, color = 'South', linetype = 'Contested'), size=1) +
  labs(x= "Date", y= "No. of Cities",
       color = 'Region', linetype = "Status", fill = "Influenced city") +
  scale_color_manual(values = colors) +
  scale_linetype_manual(values = lty) + 
  scale_fill_manual(values = fill)




