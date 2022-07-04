getwd()
setwd("~/R/Ukraine")
library(readr)
event1 <- read_csv("event1.csv")
act <- read_csv("prop.csv")

library(dplyr)
library(Rcpp)
library(tidyverse)
library(sf)
library(mapview)
mapview(event1, xcol = "longitude", ycol = "latitude", crs = 4269, grid = FALSE,
        zcol = 'region', layer.name = 'Region')
# gusess
act1 <- act %>% filter(Dates <= as.Date('2022-03-14'))
act2 <- act %>% filter(Dates > as.Date('2022-03-14'))

hist(act1$center_act - act1$east_act)
hist(act2$center_act - act2$east_act)

mean(act1$center_act)
mean(act2$center_act)
mean(act1$east_act)
mean(act2$east_act)
t.test(act1$east_act, act1$center_act)
t.test(act2$east_act, act2$center_act)

center_active <- event %>% filter(region == 'center') %>% select(date, longitude, latitude,
                                                                 t_airstrike_pred, t_armor_pred,
                                                                 t_raid_pred, t_artillery_pred,
                                                                 t_cyber_pred)


df_act <- data.frame(matrix(ncol = 3), nrow=0)
colnames(df_act) <- c('date', 'longitude', 'latitude', 'type') 
for (i in 1:nrow(center_active)) {
  if (center_active[i,4] == 1) {
    new.row <- data.frame(date = center_active[i,1], longitude = center_active[i,2], 
                          latitude = center_active[i,3], type = 'air_strike')
    df_act <- rbind(df_act, new.row)
  } else if (center_active[i,5] == 1) {
    new.row <- data.frame(date = center_active[i,1], longitude = center_active[i,2], 
                          latitude = center_active[i,3], type = 'armor')
    df_act <- rbind(df_act, new.row)
  } else if (center_active[i,6] == 1) {
    new.row <- data.frame(date = center_active[i,1], longitude = center_active[i,2], 
                          latitude = center_active[i,3], type = 'special')
    df_act <- rbind(df_act, new.row)
  } else if (center_active[i,7] == 1) {
    new.row <- data.frame(date = center_active[i,1], longitude = center_active[i,2], 
                          latitude = center_active[i,3], type = 'artillery')
    df_act <- rbind(df_act, new.row)
  } else if (center_active[i,8] == 1) {
    new.row <- data.frame(date = center_active[i,1], longitude = center_active[i,2], 
                          latitude = center_active[i,3], type = 'cyber')
    df_act <- rbind(df_act, new.row)
  }
}
df_act <- na.omit(df_act)
air <- df_act %>% filter(type == 'air_strike')
armor <- df_act %>% filter(type == 'armor')
special <- df_act %>% filter(type == 'special')
artillery <- df_act %>% filter(type == 'artillery')
cyber <- df_act %>% filter(type == 'cyber')

mapview(df_act, xcol = "longitude", ycol = "latitude", crs = 4269, grid = FALSE, zcol='type')









