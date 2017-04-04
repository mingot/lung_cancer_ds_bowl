sub1 <- fread(paste0(path_repo,"data/submissions/12_submission.csv"))
sub2 <- fread(paste0(path_repo,"data/submissions/11_submission.csv"))
setkey(sub1,id)
setkey(sub2,id)
cancer1 <- sub1$cancer
cancer2 <- sub2$cancer

subfinal <- sub1
subfinal$cancer <- (cancer1+cancer2)/2
write.csv(subfinal,paste0(path_repo,"data/submissions/13_submission.csv"), quote=F, row.names=F)
