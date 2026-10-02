install.packages("vars")
library(vars)
var_model <- VAR(data_matrix, p = 1, type = "const")
T<-Bcoef(var_model)
u <-residuals(var_model)
lengtht<-nrow(T)
I<-diag(lengtht)
sig<-.5
A <-solve(I-sig*T)
n<-5
e1<-numeric(n)
e1<-e1[1]
rnews<-e1%*%A%*%u
e2<-numeric(n)
e2<-e2[2]
mnews<-e2%*%A%*%u
e3<-numeric(n)
e3-e3[3]
rnews<-e3%*%A%*%u
e4<-numeric(n)
e4<-e4[4]
cnews<-e4%*%A%*%u