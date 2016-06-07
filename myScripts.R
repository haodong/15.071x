setwd('/Volumes/data/Documents/MOOC/edX/15.071x The Analytics Edge/Kaggle Competition/')
rawDF <- read.csv('NYTimesBlogTrain.csv', stringsAsFactors=F)
subDF <- read.csv('NYTimesBlogTest.csv')

##  basic pre-process
df <- rbind(rawDF[,-9], subDF)
df$NewsDesk <- as.factor(df$NewsDesk)
df$SectionName <- as.factor(df$SectionName)
df$SubsectionName <- as.factor(df$SubsectionName)
#df$PubTime <- gsub("^.+ (.+)$", "\\1", df$PubDate)
df$PubDate <- as.POSIXct(df$PubDate)
df$PubDay <- as.Date(df$PubDate)
df$PubMonth <- as.factor(months(df$PubDate))
df$PubWeekday <- as.factor(weekdays(df$PubDate))
df$PubHour <- as.factor(paste0("Hour.", gsub("^(.+) (.+)(:\\d{2}){2}$", "\\2", df$PubDate)))
df$WordCountLog <- log(df$WordCount+1)
df$HeadlineWC <- nchar(df$Headline)
df$AbstractWC <- nchar(df$Abstract)
library(lattice)
xyplot(HeadlineWC~AbstractWC, data=df[1:nrow(rawDF), ], group=rawDF$Popular)
xyplot(WordCountLog~AbstractWC, data=df[1:nrow(rawDF), ], group=rawDF$Popular)
str(df)
summary(df)
#rawDF$Popular <- as.factor(ifelse(rawDF$Popular==1, "yes", "no"))
rawDF <- cbind(Popular = rawDF$Popular, df[1:nrow(rawDF), c(1:3, 11:15)])
subDF <- df[(nrow(rawDF)+1):nrow(df), ]
rawDF[1,]

submitResult <- function(mod, dSet, type="response"){
    if (type=="prob") {
        subPred <- predict(mod, dSet, type=type)[,2]
    } else {subPred <- predict(mod, dSet, type=type)}
    MySubmission <- data.frame(UniqueID = dSet$UniqueID, Probability1 = subPred)
    fileName <- paste("/Volumes/RamDisk/SubmissionSimpleLog", Sys.time(), "csv", sep = ".")
    write.csv(MySubmission, fileName, row.names=FALSE)
}

##  split train set
library(randomForest)
library(rpart.plot)
library(caTools)
library(rattle)
library(caret)
library(rpart)
library(ROCR)
library(doMC)
set.seed(5)
nS <- sample.split(rawDF$Popular, .7)
tr <- rawDF[nS, ]
te <- rawDF[!nS,]
sapply(names(tr)[1:6], function(clName){rbind(table(tr[clName]), table(te[clName]))})
str(tr)
summary(tr)

##  general logistic model
mGLMrough <- glm(as.factor(Popular)~., tr, family = binomial)
summary(mGLMrough)
pGLMrough <- predict(mGLMrough, te, type="response")
confusionMatrix(ifelse(pGLMrough>=.5, 1, 0), te$Popular)$overall[1]
predGLMrough <- prediction(pGLMrough, te$Popular)
performance(predGLMrough, "auc")@y.values[[1]]
perfGLMrough <- performance(predGLMrough, "tpr", "fpr")
plot(perfGLMrough, colorize=T, print.cutoffs.at=seq(0,1,.1), text.adj=c(-.2,1.7))
##  cart model
trControl <- trainControl(method="cv", number = 10)
cpGrid <- expand.grid(.cp = (0:500)*1e-5)
trainCARTrough <- train(Popular~., data=tr, method="rpart", trControl = trControl, tuneGrid = cpGrid)
set.seed(5)
mCARTrough <- rpart(Popular~NewsDesk+SectionName+SubsectionName+WordCount+PubDay+PubMonth+PubWeekday, data=tr, method="class", cp=trainCARTrough$bestTune[[1]])
confusionMatrix(predict(mCARTrough, te, type="class"), te$Popular)$overall[1]
predCARTrough <- prediction(predict(mCARTrough, te, type="prob")[,2], te$Popular)
performance(predCARTrough, "auc")@y.values[[1]]
perfCARTrough <- performance(predCARTrough, "tpr", "fpr")
plot(perfCARTrough, colorize=T, print.cutoffs.at=seq(0,1,.1), text.adj=c(-.2,1.7))
##  random forest
registerDoMC(cores = 4)
fitControl <- trainControl(method = "cv", classProbs = TRUE, summaryFunction = twoClassSummary, allowParallel = T)
set.seed(5)
tRFrough = train(Popular~., tr, method="rf", nodesize=5, ntree=5e2, metric="ROC", trControl=fitControl, do.trace=T)
tRFrough
set.seed(5)
mRFrough <- randomForest(Popular~., data=tr, ntree=5e2, replace = F, localImp = T, mtry = tRFrough$bestTune[[1]], do.trace=100)
mRFrough
confusionMatrix(predict(mRFrough, te), te$Popular)$overall[1]
predRFrough <- prediction(predict(mRFrough, te, type="prob")[,2], te$Popular)
performance(predRFrough, "auc")@y.values[[1]]
perfRFrough <- performance(predRFrough, "tpr", "fpr")
plot(perfRFrough, colorize=T, print.cutoffs.at=seq(0,1,.1), text.adj=c(-.2,1.7))

##  hierarchical clustering
##  for headline
if (T) {
    headDist <- dist(ltH, method = "euclidean")
    headHC <- hclust(headDist, method = "ward.D")
    #plot(headHC)
    #rect.hclust(headHC, k = 3, border = "red")
    headHCluster <- cutree(headHC, k = 3)
    df$HCluster <- as.factor(paste0("H", headHCluster))
}
##  for snippet
if (F) {
    snipDist <- dist(ltS, method = "euclidean")
    snipHC <- hclust(snipDist, method = "ward.D")
    #plot(snipHC)
    #rect.hclust(snipHC, k = 4, border = "red")
    snipHCluster <- cutree(snipHC, k = 4)
    df$SCluster <- as.factor(paste0("S", snipHCluster))
}
##  for content
if (T) {
    wclDist <- dist(df$WordCountLog, method = "euclidean")
    wclHC <- hclust(wclDist, method = "ward.D")
    #plot(wclHC)
    #rect.hclust(wclHC, k = 4, border = "red")
    wclHCluster <- cutree(wclHC, k = 4)
    df$wclCluster <- as.factor(paste0("S", wclHCluster))
}
##  split
str(df)
rawDF <- cbind(Popular = rawDF$Popular, df[1:nrow(rawDF), -c(4:10)])
subDF <- df[(nrow(rawDF)+1):nrow(df), ]
tr <- rawDF[nS, ]
te <- rawDF[!nS,]
summary(tr)
tr[1,]
##  random forest
set.seed(5)
tRFhcluster = train(Popular~., tr, method="rf", nodesize=5, ntree=2e3, metric="ROC", trControl=fitControl, do.trace=T, replace = F, localImp = T)
plot(tRFhcluster)
plot(tRFhcluster$finalModel)
set.seed(5)
mRFhcluster <- randomForest(Popular~., data=tr, ntree=2e3, nodesize=5, replace = F, localImp = T, mtry = tRFhcluster$bestTune[[1]], do.trace = 100)
mRFhcluster
confusionMatrix(predict(mRFhcluster, te), te$Popular)$overall[1]
predRFhcluster <- prediction(predict(mRFhcluster, te, type="prob")[,2], te$Popular)
performance(predRFhcluster, "auc")@y.values[[1]]
perfRFhcluster <- performance(predRFhcluster, "tpr", "fpr")
plot(perfRFhcluster, colorize=T, print.cutoffs.at=seq(0,1,.1), text.adj=c(-.2,1.7))
##  submit
submitResult(mRFhcluster, subDF, "prob")

##  natural language process
library(tm)
cleanText <- function(txt, skip=F) {
    txt <- Corpus(VectorSource(txt))
    if (!(1 %in% skip)) txt <- tm_map(txt, tolower)
    if (!(2 %in% skip)) txt <- tm_map(txt, PlainTextDocument)
    if (!(3 %in% skip)) txt <- tm_map(txt, removePunctuation)
    if (!(4 %in% skip)) txt <- tm_map(txt, removeWords, stopwords("en"))
    if (!(5 %in% skip)) txt <- tm_map(txt, stemDocument)
    txt
}
##  for headline
if (T) {
    hedl <- cleanText(df$Headline)
    dtmH <- removeSparseTerms(DocumentTermMatrix(hedl), .97)
    dtmH
    ltH <- as.data.frame(as.matrix(dtmH))
    colnames(ltH) <- make.names(paste0("H.", colnames(ltH)))
    sort(colSums(ltH), decreasing = T)
}
#str(ltH)
##  for snippet
if (F) {
    snpt <- cleanText(df$Snippet)
    dtmS <- DocumentTermMatrix(snpt)
    dtmS <- removeSparseTerms(dtmS, .978)
    dtmS
    ltS <- as.data.frame(as.matrix(dtmS))
}
#colnames(ltS) <- make.names(paste0("S.", colnames(ltS)))
#str(ltS)
##  for abstract
if (T) {
    abst <- cleanText(df$Abstract)
    dtmA <- removeSparseTerms(DocumentTermMatrix(abst), .97)
    dtmA
    ltA <- as.data.frame(as.matrix(dtmA))
    colnames(ltA) <- make.names(paste0("A.", colnames(ltA)))
    sort(colSums(ltA), decreasing = T)
}
#str(ltA)
##  combine
#df2 <- cbind(df, ltH, ltS, ltA)
##  capital
df$initCharH <- as.factor(sapply(cleanText(df$Headline, 4), function(corpus){
    #corpus <- gsub("\\s+", " ", corpus)
    #corpus <- gsub("^ *(.+) *$", "\\1", corpus)
    #words <- strsplit(corpus, " ")
    #init <- words[[1]][1]
    #init <- strsplit(init, "")[[1]]
    #init[1]
    strsplit(gsub(" ", "", corpus[[1]][[1]]), "")[[1]][1]
}))
table(df$initCharH)
##  split again
df2 <- cbind(df, ltH, ltA)
rawDF <- cbind(Popular = rawDF$Popular, df2[1:nrow(rawDF), -c(4:10)])
subDF <- df2[(nrow(rawDF)+1):nrow(df), ]
str(rawDF)
summary(rawDF)
rawDF[1,]
tr <- rawDF[nS, ]
te <- rawDF[!nS,]
##  cart model
registerDoMC(cores = 4)
trControl <- trainControl(method="cv", number = 10, allowParallel = T)
cpGrid <- expand.grid(.cp = seq(1e-3, 3e-3, 2e-5))
set.seed(5)
tCARTnlp <- train(Popular~., data=tr, method="rpart", trControl = trControl, tuneGrid = cpGrid)
tCARTnlp
set.seed(5)
mCARTnlp <- rpart(Popular~., data=tr, method="class", cp=tCARTnlp$bestTune[[1]])
confusionMatrix(predict(mCARTnlp, te, type="class"), te$Popular)$overall[1]
predCARTnlp <- prediction(predict(mCARTnlp, te, type="prob")[,2], te$Popular)
performance(predCARTnlp, "auc")@y.values[[1]]
perfCARTnlp <- performance(predCARTnlp, "tpr", "fpr")
plot(perfCARTnlp, colorize=T, print.cutoffs.at=seq(0,1,.1), text.adj=c(-.2,1.7))
##  random forest
fitControl <- trainControl(method = "cv", classProbs = TRUE, summaryFunction = twoClassSummary, allowParallel = T)
set.seed(5)
tRFnlp = train(Popular~., tr, method="rf", metric="ROC", trControl=fitControl, ntree=400, nodesize=5,
               replace = F, localImp = T, proximity = F, do.trace = 100)
tRFnlp
plot(tRFnlp)
plot(tRFnlp$finalModel)
mRFnlp <- foreach(nTree=rep(100, 4), .combine = combine, .multicombine = T, .packages = "randomForest") %dopar%{
    set.seed(5)
    randomForest(Popular~., data=tr, ntree=nTree, nodesize=5,
                 replace = F, localImp = T, proximity = F, mtry = tRFnlp$bestTune[[1]], do.trace = 10)
}
set.seed(5)
mRFnlp <- randomForest(factor(Popular)~., data=tr, ntree=400, nodesize=5, replace = F, localImp = T, proximity = F, mtry = tRFnlp$bestTune[[1]], do.trace = 100)
sapply(1:5, function(nodeS) {
    mRFnlp <- randomForest(factor(Popular)~., data=tr, ntree=400, nodesize=nodeS, replace = F, localImp = T, proximity = F, do.trace = 100)
    print(mRFnlp)
    print(confusionMatrix(predict(mRFnlp, te), te$Popular)$overall[1])
    predRFnlp <- prediction(predict(mRFnlp, te, type="prob")[,2], te$Popular)
    print(performance(predRFnlp, "auc")@y.values[[1]])
})
mRFnlp
plot(mRFnlp)
confusionMatrix(predict(mRFnlp, te), te$Popular)$overall[1]
predRFnlp <- prediction(predict(mRFnlp, te, type="prob")[,2], te$Popular)
performance(predRFnlp, "auc")@y.values[[1]]
perfRFnlp <- performance(predRFnlp, "tpr", "fpr")
plot(perfRFnlp, colorize=T, print.cutoffs.at=seq(0,1,.1), text.adj=c(-.2,1.7))
##  submit
submitResult(tRFnlp, subDF, "prob")

##  multi-level factors to multi variables
lev2var <- function(dataset, colName) {
    levName <- levels(dataset[, colName])
    levName <- gsub("[ /&]", ".", levName)
    for (newVar in levName) {
        if (newVar=="") {
            dataset[,paste(colName, "NA", sep = ".")] <- dataset[,colName]==""
        } else dataset[,paste(colName, newVar, sep = ".")] <- dataset[,colName]==newVar
    }
    colPos <- which(colnames(dataset) %in% colName)
    dataset[,-colPos]
}
Lev2Var <- function(dataset) {
    candiList <- names(dataset[sapply(sapply(subset(dataset, sapply(dataset, class)=="factor"),levels), length) > 2])
    for (colName in candiList) {
        dataset <- lev2var(dataset, colName)
    }
    dataset
}
df3 <- Lev2Var(df2)
any(is.na(df3))
##  split again
rawDF <- cbind(Popular = rawDF$Popular, df3[1:nrow(rawDF), -c(1:7)])
subDF <- df3[(nrow(rawDF)+1):nrow(df3), ]
str(rawDF)
summary(rawDF)
rawDF[1,]
tr <- rawDF[nS, ]
te <- rawDF[!nS,]
##  random forest
registerDoMC(cores = 4)
fitControl <- trainControl(method = "cv", classProbs = TRUE, summaryFunction = twoClassSummary, allowParallel = T)
set.seed(5)
tRFmlf2v = train(Popular~., tr, method="rf", metric="ROC", trControl=fitControl, ntree=2000, nodesize=5,
                 replace = F, localImp = T, proximity = F, do.trace = 100)
tRFmlf2v
plot(tRFmlf2v)
tRFmlf2v$finalModel
confusionMatrix(predict(tRFmlf2v, te), te$Popular)$overall[1]
rattle(tRFmlf2v$finalModel)
mRFmlf2v <- foreach(nTree=rep(500, 4), .combine = combine, .multicombine = T, .packages = "randomForest") %dopar%{
    set.seed(5)
    randomForest(Popular~., data=tr, ntree=nTree, nodesize=5,
                 replace = F, localImp = T, proximity = F, mtry = tRFmlf2v$bestTune[[1]], do.trace = 10)
}
set.seed(5)
mRFmlf2v <- randomForest(Popular~., data=tr, ntree=400, nodesize=5, replace = F, localImp = T, proximity = F, mtry = tRFmlf2v$bestTune[[1]], do.trace = 100)
mRFmlf2v
plot(mRFmlf2v)
confusionMatrix(predict(mRFmlf2v, te), te$Popular)$overall[1]
predRFmlf2v <- prediction(predict(mRFmlf2v, te, type="prob")[,2], te$Popular)
performance(predRFmlf2v, "auc")@y.values[[1]]
perfRFmlf2v <- performance(predRFmlf2v, "tpr", "fpr")
plot(perfRFmlf2v, colorize=T, print.cutoffs.at=seq(0,1,.1), text.adj=c(-.2,1.7))
##  submit
submitResult(mRFmlf2v, subDF, "prob")
