#dev.off(dev.list()["RStudioGD"])

rm(list=ls())
library(tidyr)
library(tidyselect)
library(readr)
library(ggplot2)
library(forcats)
library(rlang)
library(dplyr)
theme_set(theme_bw())
allParams <- read_csv("/home/frederik/mnt/ATIAjbv/Results/AllResults.csv")

allParams <- allParams %>% mutate(Dataset = recode(Dataset, "Melanoma" = "ISIC", "PNEUMONIA" = "Pneumonia"), 
                                  Modeltype  =recode(Modeltype, "BASECASE" = "Basecase", "4X4" = "4x4", "PATCH"="Patch"))

allParams$Modeltype <-as_factor(allParams$Modeltype)
levels(allParams$Modeltype) = c("Basecase","4x4","Patch")
ds <- allParams
ggplot(data=ds, mapping=aes(col=Modeltype, y = AUC)) + geom_boxplot()  + facet_wrap(~Dataset)+ 
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + ylab("AUC-score")
ggsave(filename="/home/frederik/Documents/ATIA/Results/AUCmodel.png", device="png")
ggplot(data=ds, mapping=aes(col=Modeltype, y = ACC)) + geom_boxplot() + facet_wrap(~Dataset) + 
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + ylab("Accuracy (%)")

ggsave(filename="/home/frederik/Documents/ATIA/Results/ACCmodel.png", device="png")
ggplot(data=ds, mapping=aes(col=Modeltype, y = EPOCH)) + geom_boxplot() + facet_wrap(~Dataset) + 
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) + ylab("Number of epochs to converge")
ggsave(filename="/home/frederik/Documents/ATIA/Results/EPOCHmodel.png", device="png")

dss = c("ISIC", "Pneumonia")
AUCm = matrix(rep(NA,length(dplyr::filter(ds, Dataset=="ISIC")$AUC)), ncol = 3)
ACCm = matrix(rep(NA,length(dplyr::filter(ds, Dataset=="ISIC")$AUC)), ncol = 3)
EPOCHm = matrix(rep(NA,length(dplyr::filter(ds, Dataset=="ISIC")$AUC)), ncol = 3)
for (el in dss){
  for (i in 1:3){
  AUCm[,i] <- dplyr::filter(ds, Dataset=={{el}}&Modeltype==levels(Modeltype)[{{i}}])$AUC
  ACCm[,i] <- dplyr::filter(ds, Dataset=={{el}}&Modeltype==levels(Modeltype)[{{i}}])$ACC
  EPOCHm[,i] <-dplyr::filter(ds, Dataset=={{el}}&Modeltype==levels(Modeltype)[{{i}}])$EPOCH
  }
  ms = apply(AUCm, 2, mean)
  sds = apply(AUCm,2, sd)/sqrt(10)
  cat(paste("In the, ",el, "dataset, the AUCmeans are:", ms[1], ms[2], ms[3], ", \nwhile sderror are:", sds[1], sds[2], sds[3], "\n"))
  ms = apply(ACCm, 2, mean)
  sds = apply(ACCm,2, sd)/sqrt(10)
  cat(paste("In the, ",el, "dataset, the ACCmeans are:", ms[1], ms[2], ms[3], "\nwhile sderror are:", sds[1], sds[2], sds[3], "\n"))
  ms = apply(EPOCHm, 2, mean)
  sds = apply(EPOCHm,2, sd)/sqrt(10)
  cat(paste("In the, ",el, "dataset, the EPOCHmeans are:", ms[1], ms[2], ms[3], "\nwhile sderror are:", sds[1], sds[2], sds[3], "\n\n\n"))
}

flops <- "GLOPS: /n BASECASE: 9.13 /n 4X4: 8.97/n PATCH: 8.91 GFLOPs /n"
