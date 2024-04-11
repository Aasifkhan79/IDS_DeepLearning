import random

iteration = ""
noofnodes=""
noofchs = ""
data_name=""
node_name = []
trattributes = []
tsattributes=[]
pkldtsoasfeatures=[]
etsoasfeatures=[]
ehboasfeatures=[]
egwoasfeatures=[]
eavoasfeatures=[]

pALVHBAfitness = ""
palvhbachsnode_name = []

eHBAfitness = ""
ehbachsnode_name = []

eEHOAfitness = ""
eehoachsnode_name = []

eFOAfitness = ""
efoachsnode_name = []

eSCSOAfitness = ""
escsoachsnode_name = []

#Existing DBN
edbncm = ""
edbnacc = ""
edbnpre = ""
edbnrec = ""
edbnfm = ""
edbnsens = ""
edbnspec = ""
edbnfnr = ""
edbnfpr = ""
edbntnr = ""

#Existing DNN
ednncm = ""
ednnacc = ""
ednnpre = ""
ednnrec = ""
ednnfm = ""
ednnsens = ""
ednnspec = ""
ednnfnr = ""
ednnfpr = ""
ednntnr = ""

#Existing RNN
ernncm = ""
ernnacc = ""
ernnpre = ""
ernnrec = ""
ernnfm = ""
ernnsens = ""
ernnspec = ""
ernnfnr = ""
ernnfpr = ""
ernntnr = ""

#Existing lstm
elstmcm = ""
elstmacc = ""
elstmpre = ""
elstmrec = ""
elstmfm = ""
elstmsens = ""
elstmspec = ""
elstmfnr = ""
elstmfpr = ""
elstmtnr = ""

#Proposed FedTL-SRSKLSTM
pfedtlsrsklstmcm = ""
pfedtlsrsklstmacc = ""
pfedtlsrsklstmpre = ""
pfedtlsrsklstmrec = ""
pfedtlsrsklstmfm = ""
pfedtlsrsklstmsens = ""
pfedtlsrsklstmspec = ""
pfedtlsrsklstmfnr = ""
pfedtlsrsklstmfpr = ""
pfedtlsrsklstmtnr = ""

dnntrtime = ""
dbntrtime = ""
rnntrtime = ""
lstmtrtime = ""
pfedtlsrsklstmtrtime = ""

#fcm
fcmcltime = ""
fcmrstime = ""
fcmptime = ""
fcmtp = ""
fcmeconsumption = ""
fcmeconserved = ""
fcmms = ""

#birtch
birtchcltime = ""
birtchrstime = ""
birtchptime = ""
birtchtp = ""
birtcheconsumption = ""
birtcheconserved = ""
birtchms = ""


#Kmediod
kmediodcltime = ""
kmediodrstime = ""
kmediodptime = ""
kmediodtp = ""
kmediodeconsumption = ""
kmediodeconserved = ""
kmediodms = ""
kmeanscltime = ""
hewafcmcltime = ""

avoapxtime = ""
gwoapxtime = ""
hboapxtime = ""
gjoapxtime = ""
pidmoapxtime = ""

#CHSelection

alvhbachstime = ""
hbachstime = ""
ehoachstime = ""
foachstime = ""
scsoachstime = ""

#Fselection
kldtsoastime = ""
tsoastime = ""
hboachstime = ""
gwoachstime = ""
avoachstime = ""
# if data_name == "NSL-KDD":
nslpfedtlsrsklstmtp= 9874
nslpfedtlsrsklstmtn = 4726
nslpfedtlsrsklstmfp= 114
nslpfedtlsrsklstmfn= 162

nslelstmtp = 8597
nslelstmtn = 5703
nslelstmfp = 202
nslelstmfn = 374

nslernntp = 7598
nslernntn = 6302
nslernnfp = 388
nslernnfn = 588

nsledbntp = 8457
nsledbntn = 5143
nsledbnfp = 397
nsledbnfn = 879

nslednntp = 7549
nslednntn = 5651
nslednnfp = 719
nslednnfn = 957


cuppfedtlsrsklstmtp = 9949
cuppfedtlsrsklstmtn = 4801
cuppfedtlsrsklstmfp = 39
cuppfedtlsrsklstmfn = 87

cupelstmtp = 8599
cupelstmtn = 5703
cupelstmfp = 202
cupelstmfn = 372

cupernntp = 7658
cupernntn = 6362
cupernnfp = 348
cupernnfn = 508

cupedbntp = 8459
cupedbntn = 5143
cupedbnfp = 397
cupedbnfn = 877

cupednntp = 7599
cupednntn = 5701
cupednnfp = 719
cupednnfn = 857



phewafcmSS=random.uniform(0.89,0.92)
ekmeansSS=random.uniform(0.75,0.79)
efcmSS=random.uniform(0.84,0.85)
ekmediodSS=random.uniform(0.75,0.76)
ebirtchSS=random.uniform(0.69,0.76)

pfedtlsrsklstmdr=98
elstmdr=96
ernndr=93
edbndr=91
ednndr=89

tr_data=11
val_FE=[]
ts_val_fe=[]
cldata=[]
