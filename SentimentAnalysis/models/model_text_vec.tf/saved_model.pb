??
? ?
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????"serve*2.8.22v2.8.2-0-g2ea19cbb5758͋
m

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name57752*
value_dtype0	
?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_55227*
value_dtype0	
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
??
Const_4Const*
_output_shapes	
:?>*
dtype0*??
value??B???>BiBimBgoBgetBdayBgoodBworkBlikeBloveBtodayBtimeBcantBgotBuBthankBlolBmissBwantBbackBoneBknowBthinkBfeelBseeBitBrealliBwellBhopeBwatchBstillBnightBneedBmakeBthatBnewBhomeBlookBohBcomeBmuchB2BlastBtwitterBmornBwishBsadBtomorrowBgreatBmyBwaitBillBtheBsleepBhahaB3BhappiBjustBbadBtriBfunBrightBweekBfollowBthingBfriendBwouldBsorriBtonightByouBsayBwayBtakeBthoughBniceBdontBgonnaBbetterBevenBhateByeahBbedBsoBstartBcouldBtweetBplayBandBpeoplBhourBshowBivBschoolBwhatBbutBguyBweekendByesBheyBfinalBnextBletBuseBawesomBnoBsoonBneverBtireBfirstBlongBlittlBeveryonBbestBrainBmoviBpleaBwannaB4ByearBsickBlifeBokBsuckBgirlBfindBcallBsureBhelpBdoneBheadBboreBtalkBalwayBalreadiBnotBkeepBheBcoolBanothBsomethBaBleavBliveBlotBhurtBeatBmanBphoneBenjoyBreadiBreadByayBmadeBfaceBbigBisBhaveBhousByetBwentBsongBsoundBxBprettiBthoughtBeverBmaybBurBamazBexcitBguessB1BawayBfinishBtellBhowBsummerBdamnBomgBgameBlistenBoldBsomeonBearliBmeanBnowBgiveBleftBbabiBbitBlostBcheckBhearBweBnothBwowBendBthereBpartiBthisBlateBgladBhotBactualBhappenBpicBlaterBalsoBstopBhardBcriBbirthdayBwonderBweatherBsunBputBmomBstuffBtwoBughBnByaBwBsawBstayBmightBexamBgodB	yesterdayBrunBworldByourBsincBcarBkidBsaidBmeetBfuckBmusicBjobBbeautiBlaughBhiBupdatBgottaBsundayBfridayBseemBaroundBidBannoyBmaniBvideoBmondayBpostBluckBsmirkBcoldBfoundBatBpoorB5BwinkBmustBwhiBmoveBawwBgoneBdieBdidntBbookBboyBifBmayBplanBinBbuyBrBmeBanythBleastBbusiBfamiliBwokeBokayBshopBtotalBalmostBcuteBmonthBstudiBfoodBlunchBpBiphonBthoBtillBdinnerBwelcomBdrinkBbelievBfarBchangBcausBsweetBhairBpicturBplaceBeverythBonBwinBfreeBfunniBclassBanyonBshitBmineBturnBforwardBhaBwithoutBsitBaskBwalkBeveriBdriveBideaBstupidBhahahaBsendBnameBdadBdreamBoutsidBbBenoughBtoBallBwriteBrealBcleanBwrongBanymorBwerBcoffeBwakeBroomBhellBsaturdayBprobablBdogBawBfanBminutBheadachBpersonBmoneyBsmileyBsoooBelBseriousBtvBremembBxxBbreakBfailBseenBhitBbrotherBhangBwholeBcameB10BrepliBrockBopenBkindaBtrainBblogBbeachBcraziBeyeBrestBcloseBtheyBkillBmotherBpainBhadBtextBdoBwordBquitBcomputB	goodnightBworriBtookBhelloBanywayBupBtheyrBablBcareBsuperBtrueBemailBuneasiBundecidBproblemBbringBhesitBtripBforgotBhalfBagoBpartBkindBnewsBeitherBmindBheartBofficBphotoBwontBskepticBlinkB6BwillBfullBpayBsisterBawwwBhugBahBbooBsheBcuzBalonBinternetBheheBfallBdancBsooBcoursB8BsometimBbtwBheardBstuckBticketBpickBtestBcBwearBinterestBpasBsiteBÂ½BsetBsupposBvisitBoffBlearnBconcertBonlinBfixBdudeBaddBshowerBÂ¿BawakBfineBsurprisBtoldBseasonBtooBhandBfavoritB	breakfastBgoeBglassBfacebookBcatchBbrokeBtilBvoteBsunniBlmaoBÃ¢BsmileBiceBhighBboughtBspendBpackBasBjuneBcutBcrapBnanBdefinitBcatBwedBforBluckiBagreBmadBasleepB	afternoonB30BsignBladiBinsteadB100BhungriBreasonB7BwhereBrideBlaByeaBwhoBstoriBredBperfectBboutBwaBjealousBshortBfigurBamBgraduatBjoinBsighBsoreBleBteaBnapBcitiBsooooBsecondBtopBdeadBcongratBtogethBdearBtourBpageBmessagBhomeworkBbdayBcompletBnearBalbumBsaveBcouplB12BawardBrelaxBparkBdateBsingBstoreBholidayBpointBwaterBstarBlistBlaptopBmomentBdropBgoinBrevisBfootByoutubBveriBdownloadBcanBshareBbeB0BipodBtownBdressBdoesntBsideBchurchBdidBanswerBdecidBaccountBofficiBcookBareBÂBlineBloseBweirdBstickBscareBforgetBorderBpplBwhenBrealizBbyeBmoodBminBiveBairBhaventBgymBlilBunfortunByallBniteBfbB
understandBateB20BchatBusualBknewBenglishBahhBbandBpastBcreamBpoolBuploadBepisodBfliBmumBmathBdifferBcommentBworstBhorriblBsupportBloadBchancBfastBchocolBcheerBkickBlondonBaintBhmmBthroatBworthBthenBteamBwindowBofBflightBupsetBstraightBonliBparentBthreeBsatBquestionBblowBbrokenByepBsleepiBblessBdepressBblackB1stBviaBmrBsunshinBcardBslowBxxxBsentBgaveBvBlegBfairBspecialBbeatBfollowfridayBcollegBnumberBwebsitBmacBjonaBgrinBbetBapparBadBfingerBrecordBtuesdayBfellBwarmBgreenBmoonBdBemBfilmBratherBdaBworsBvacatBpaperBeBlongerBpossiblB
disappointBwithBprojectBspentBlightBgardenBdueBappB9BnopeBhereBblueBpowerBjuliBfreakBwtfBabsolutBcannotBbeerBoBeasiBsimBshameBbodiBcakeBearlierBmessBcelebrBsafeBmileyBhugeBplusBlayBcousinBchillBwomanBbabeBstressBageBbikeBisntBmtvBgooglBfluBquickBthxBremindBthursdayBlaziBahhhBcancelBburnBforevB15BcoBvoicBexceptBkBshoeBespeciBexactBshootBshotBswimBappreciBoutBholdBstomachBcampB	boyfriendBsortBwhiteBtouchBcdBbusBmetBsonBbumBmanagBorBcameraBsleptBcurrentBidkBpissBlieBprayBmBfightBdavidBcrashBkissBukBmyspacBtypeBroadBpresentBdmBapplBinsidByummiBinvitBtomBterriblBcoverBbitchBsiBmeantBitllBarrivBxdBwitB	interviewB2dayBfatherBairportBboxBluvBtummiBblockBtastBchickenBhospitBsmallBrandomBrainiBcaseBconfusBexpectBclubBhrBbbqBnoteBservicBfitBradioBcountBnoticBclothBhubbiBbeginBdealBdoctorBhillBalrightBstormBproudBshirtBdesignBsmellBmoreBspeakBpizzaBfactBmentionBsearchBfeltBfrontBstandBgoodbyBaddictBwineBloneBtwilightBshallBpullBachBwashBdvdB	wednesdayBbeenBcompaniBtearBissuBproductBpracticBhmmmByupBwhatevBcupBconnectBpeacBgorgeousBlakerBxoxoBexhaustBsBotherBtakenBlBfrenchBlameBouchBbagBmemoriBaboutBeventBwooBrollBpmBapartBbarBsoldBbecomBstateBsellBbroB	everybodiBchildBallowByrBoncBdaughterBjokeBgettinB2ndBscariBripBfantastBbehindBtanBprobBhangovBnormalBdrunkBarmBearBcouldntBtravelBreturnBmommiBpaintBmatterBalthoughBsinglBÂBreleasB11BplaneBpuppiBdarkBalongBÂ´BfireBvegaBguitarBmateBwifeBversionBroundBclearBpromisBmileBmagicBballBdoorBalotBsaleBjusBcrossBcheekiBartBactBhahahBcountriBgroupBsomeBruinBpopBfriBfBvipBohhBwatBcookiBdeathBhuhBwastBtrackBindeBprofilBgoshBhotelBdeservBagainBwebBhistoriBinstalBaheadBhunBperformBbuddiBheatBfishBangelBraceBscreenBbugBnickBfmlBextraBehByoBpreparBpcBthrowBdeletBbbBmajorBgutBtroublBsuggestB	recommendBfillBtwitBnobodiBgrowBbloodiBdefBdaddiBbirdBtrafficBitunBffBlandBstreetBnoseBlowBcaughtBjBchillinBtaylorBranBmallBmailBviewBresultBfatBwatchinBreportBinspirBdunnoBpositBnailB	raspberriBcheesBsexiBnycB	congratulBtBhilariBdangBshutBsomewherBfavBgayBcozBvidBdoubtBdirectBskyBshineBbloodBoohBjoeBfamBmarkBaniBmostBpinkBhusbandBshouldBchoicBeditBafterBmarketBexperiB	tweetdeckBcoughBbillBsilliBwasntBlessonBinfoBjohnBchannelBcodeBbrainBreviewBsoooooBjonBÃ¯BÂ«BfeverBarghBfuturBselfBnoneBshoutBringBnahBanimByoungBadorBmmmBcopiByumBmixBquietBproperBpeepBcolorBblastBbankBteethB
blackberriBkateBdemiBstepBÂ¥BbuildB
backgroundBavailBkittiBgigBfreshBbummerBtrailerBdoublBdoinBofferBnyBnaBetcBpiecBtipBwetBseriBimaginBcertainBshiftBspotBoftenBthruBbathBawwwwBmenBjoyBepicBmiddlBjumpBpriceBbreathB24BtrekBscrewBamericanBmatchBdegreBmcfliBimportBdeBincludBhahahahaBfolkBadamBtuneBfeedBcontactBhoneyBburntBboardBfrancBtalentBwarBsecretBdoeBstarbuckBlaundriBchrisBdentistB50BimaBcrackBextremBsweetiBmarriBsanBfavouritBsmokeBgiftBwallBourBfeelinB	australiaBkingBexpensBbotherBschedulBsenBmouthBgBsuccessBkeyBsouthBpaidBconsidBhonestBshesBareaBblahBchicagoBbuttonBsessionBtopicBrealisBfaveBbunchBlimitBsumBcontinuB25BdarnBcellBlipBfourBmsnBnervousBseatBtweepBentirBbatteriBthanxBquotBafraidBexcelBkneeBignorBteacherBeggBpromBarticlBoriginBimpressBclickBchipBbfBspellBsomebodiBcreatBbareBunlessBmamaBpushBreceivBmilkBjackBplzBfloorBhowevBhBaffordBspaceBbottlBmobilBtonBb4BgottenBbiteBupgradBstationBnoonBuhBsunburnBrobBswineBsocialBdriBstudioBmidnightBeverydayBanywherBmachinBassignBgrandmaBgermaniBlocalBdntBhannahB3rdBwaveBdeliciBbosBworkoutBrowBattackBtreeBknoBfrustratBxoBreachBfreakinBtrendBteachBsystemB	entertainBlakeBdrawBstudentBthunderBholiBloudB40BoopBdanniBwevBdrBcustomBhehehBwoohooBdragBtonitBheroBenergiBtwiceBbowlBtreatBouttaBfileBislandBcrappiBfiveBessayBserverBbrightBbrazilBspamB2009BclientBÂ£BmediumBharriBchargBfestivBrecentBprayerBemptiBtrustBneckBumBsizeBreplacBlibrariBaugustBamericaBforcBcorrectBretweetBbakeBgunnaBbroughtBmainB	everywherBstyleBspringBsexBruleBwroteBcharactBkitchenBplayerBcontrolBcraveBlackBprogramBalivBsushiBboatBjamBdisneyBflatBswearBstrangBfromBsciencBenglandB
girlfriendBnetworkBcostB18B13BimmaBdietBxboxBjbB	brilliantBaccidBscoreBuniversBflowerBdelayBpublicBslightBtoughBcouchB14BplsBfeaturBbuttBrelatBtoeBsportBgradBappearBswitchBuniBpreferBtwitpicBstrongBbearBdatByorkBjkBfabulBweightBlevelBstageBsimplBlossBcloudBsoupBattendBwestBpantBooohBcuddlBattemptBwootBstBresponsBobviousBlockB16BperhapBformBlogBbffBmillionBsuddenBscreamBfakeBdramaBconversBohhhBmoBdammitBsamBbcBoptionBahhhhB17BugliBcanadaBmarathonBgradeBfreezBchineByBkeptBblameBbgtBroughBactivBpariBmorninBlawB
strawberriBincredBshakeBalexBpleasurBhavBdreadBcreditB21BmattBjuicBwiiBskinBeasierBatmBsufferBiliBrespondBallergiBhumanBhahBgrrrBwingBsomehowBnationBnightmarBtablBsurvivBsirBorangBlordBsuchB	transformBsoccerBrequestBtweeterBphysicBnaturBdevelopB80BmiBoverBmikeBshipBjameBfloridaBacceptBseverBevilBshockBdumbB	difficultBbostonBgahBdeskBbrownB2morrowBimagBdcBmichaelBhideBgivenBexplainBacrossBminiBgreyB4thBhooBaccessBanytimBwinterBontoBnoooBsurgeriBproBfaultBspanishBfootbalBpairBmodelBsandwichBsnowBanBbesidBtruliBpetBtexaB
squarespacBsixBmmmmBworkinBobBplantBhehBeffectBshittiBndBdetailB09BhandlBnorthBkatiBwindBrealitiBlookinBangriBprintBmacbookBtattooBnastiBhatBoddBironBcontestBbasicBhookBbecausBsuitBlatestBbootBliterBhostBadvicByoullBandiBrentBmassivBinfectBbyBscratchBauntBnetBtoastBrecovBtonguBremovBnooooBbreadBhealthiBe3BryanBdailiBcarriBstatusBgolfBcupcakBughhBtoniBpourBleadBcrushBletterBfuckinBtenBanybodiBstreamBsmhBbiggestBbestiBalarmBuserBsaBbrandBdiverBcoastBsarahBexcusBawhilBunclBlmfaoBcheapBthaBshoulderBerrorBÂ°BsmartBmealBconferBfabBdriverBdocBcablBattentBaweBaimBtacoBgoodmornBcompBchoosBtweetiBsceneBgrossBsameBknockBgrandBchickBahaBhorsBtoyBotherwisBbraceBherBgeneralBconanBheavenBthemeBsusanBgroundBcollectBearthBcomplainBholeBdiscovBcaliBtshirtBsoulBlaunchBasapBabtBmotivBdaveBmemberBvetBridiculBsaladBheckBbasebalBsugarBannouncBpaBgrrBfocusBspreadB2niteBtheyllBjayBheaviBclassicBmistakBinsanBfearBeverytimBtruckBridBkillerBnutBinformBgrabBthirdBenterBanklBjeanBbenB3dBqueenBbaseB45BwouldntBdirtiByardBspeedBlaurenBseniorBemotBskipBcloserBthBshellBarentBhuntBpenBolderBbellBÂ»BprocessBunitBsooooooBrushBrestaurBmiserBexistBashleyB140BtxtBterminBqualitiBwhichBlovinBhellaBcreativBidolB
californiaBsuxBhurriBdistractBtermBlooBknownBcloudiBchallengBeuropBsydneyBresearchBdeepBfellowB1000BwifiBappliBpoutBneighborBgroceriBmontanaBjesusBdesperBbabysitB90BtixBgermanBÂ©BthtBmigrainBadmitBtenniBÂBidiotBactionBweeBtightBngBmedBjessBedBartistBthankyouBregularBkevinBdecisBconfirmBadventurBfamousBatlBcomfortBpressBeastBdecentBcandiBtagBrateBeatenBdependBtiniBbangBhaircutBsoftwarBfedBdollarBneitherBpancakBcurBbananaBalcoholBmariBflashBdowntownBcolourBbustBrepeatBloserBchuckBslowliBmehBerBavoidBgoalBstokeBkittenBfashionBthunderstormBmaxBjustinBfrownBbritneyBquotwinkBunfollowBwideBregretBhavinBgasBpoundBpaulB200BÂ¼BtruthBforumBooBgreetBepBcommerciBbedtimBahahaBorganBcenterBbaconBappointBmissinBgrillBexercisBcornerBcaBboooBbeeBtyBmountainByahBtechBreferBimprovBughhhBgearBburgerBorlandoBtryinBcancerBbornBgeekBdishBwhenevBtoothBpotterBbathroomBkelliBcashBthoseBsettlBbiggerBmuseumBmmBgnightBftwBfieldBchairB22B	disappearBcrowdBbayBrubbishBcavByouuBtwinBnephewBanniversariBmakeupBinvolvBebayBsituatBhealthBbunniBseaBriteBriceB	philippinBfanciBfruitBmiamiBÂ¸BperiodBÂ¡BpubBmagazinBrehearsBkeyboardBbutterBavatarBpieBstarvBbottomBaddressBwisdomBswiftBrubBinternB3gBmassagBclockBboneBrootBitaliBdiscusBwrapBdonutBtiBcokeBskypeBcuriousBcrewBusaBguiltiBfasterBriseBbelliBwinnerBummBthumbBgossipBewBÂ³BxxxxBpreBdallaBbetaBbeanBÃ£BmodeBlanguagBamountBacByehBpromotBgiantBcastB5amBniecBdownBstockBspiderBrefreshBcapBbuckBflipBseattlBtmrwBjasonBiranB33BdjBwickBstinkBhackBdollBbombBamiBwoooBtitlBhayfevBmsgBboylBspainBpumpBpoopBclimbBatleastBstealBparadBbubblBwhoaBhmBexBÂ±BwildBjerseyBhisBsomedayBservBpitiBnoisBjoshB6amBwenBtoolBdisBcafeB60BwarnBnowherBhwBdanBfirefoxBcruisBprincessBloverBdoggiBcontentBclueB23BweakBsteveBpeterBbahBtoooBgr8BdarlBapplicB5thBplentiBdemonBcomicB500BoooBguestBkoBheyiBgainBchapterBandriBÂBwouldvBsoakBsnapBroseBriverBpurplBdataBwithinBtorontoBrefusBpalmBiÃ¢BgoldBtieBpurchasBdigB19BmeeBtimBhopBstoleBconvincByellBcinemaBstoneBcyrusBalBrudeBmonkeyBtargetBprivatBalaBÂ¾BoceanBlyricBrayBnooBapologBzooBtommorowBthrewBsnackBraisBnothinBlargBharderBsfBperBgagaBamusByellowBlabBbeyondB4amBwrittenBurghBtradeBspiritBnoodlBf1BskillBmommaBwhilstBdutiBdelivBbruisBshapeBreBpianoBphilliBowBmusclB35BvanBtalkinBgreatestBblinkBmonsterBthinkinBsectionBfoolBÂ²BswollenBqBpicnicBimpossBiiBclipBwhetherBjordanBcrampB70BpureBcompetitByouvBdroveBuntilBopinionBfoxBdrugByuckBwolverinBgeorgB	chemistriByogaBwokenBtheyvBrobertBishBtapeBsneezBpotatoBpooBdesktopBtheseBelectrBheelBqueBarBtaBnerdBmcdonaldBwoopBpackagBmissionBlocatBhonorBpremierBhimBgooodBbioBtrickBchestBchaseBfunctionBjimmiBfaithBcerealBcreepiBstalkBroflBrequirBposterBoutfitBjourneyB7amB300BsweatBinsomniaBfulliBchargerBthnxBolBmasterBghostB	basketbalBvirusBprogressBnakeBÃ°BmexicoBcominBbattlBsealBrespectBfunerByeyBsurfBseptembBgaragByikeBhungBhipBdeckBakaBpodcastBmousBhdBsingerBpunchBhumidBgfBdrankBchelseaBcelebB0amBÃ±BspoilBgrassBawsomB3amBjapanesBhayBconBblondBaptBpigBjapanBhealBmeltBfrmBstrugglBhungovBaswelBthurBsecurBiconBcureBpotBmetroBitemBdespitBsimilarBnursB	embarrassBdefinBbullBbobB360BwereBplugBhomiBbingBaustinBaccordBtemptBstareBsimpliBhoustonBcopBawhBprolliBerrandBcomBassumBwalmartBsittinBedwardBrecipBroastBirelandBcheatBzombiByawnBspinBindiaBpretendBexpressBwhoopBsriBsockBrichBmonB	communitiBgrrrrBemiliBcontractBangB2moroBurlBthingiBsquarBpeanutBauditBaprilBaceBrareBmanchestBrelationshipBliftBkimBcyclBcoachBpopularBpatientBlightnBicecreamBgraphicBitdBitalianB	iranelectBprovidBmentalBbtBbrowniBtwitterversBsubjectBscriptBpillBkrisBcharmBbooooBwoodBweedBgrownBduckBcalmBsplitBpreviousBpillowBlawnBhoBÂBquizBomfgBfemalBbrunchBcomparBsteakBoilBgunBdigitBfreedomBenginBbbcB6thBÂ¬Bps3BlolzBchinaBpolitBmexicanBfarrahBcouldvBspeechBskoolBownBmakinBdestroyBchBbritainBbkBtomorowBpreviewBpokerBpastaBmassBcamBapprovB3gsBgloriousBgdBaccidentBsaucBhawaiiBtowardBsmBshaveBregistBhireBforgivBfeedbackB
complimentBcntBwanaBstufB
procrastinBflagBbumpByeahhBsingaporBjogBwriterBjudgBatlantaBtankBhallB	forgottenBearnBborrowBummmBtomatoBteenBtaxBsunburntBgirBbritishBÂ¦BofflinBprideBholliBdamnitBcurlBbuggerBwheelBrepairBpressurBcourtB	technologB	photoshopB	neighbourBjacketBintensBfridgBefBdonatBcommonBbudB
photographBjeffBenBcutestBsmashBnoooooBmiaBjerkB	hollywoodBespBescapBcuBcherriBcharliByuBtheaterBsnugglBsneakBselenaBnonBlongestBtranslatBsubwayBskateB	microsoftBdawnBtweeplBnoeBoctobBheeBeffortBplayinBopportunBgloomiBeasiliBcabBbleedBgrandpaBbothBallergBprotectBmuffinBhmmmmBandrewBmeatBloungBashBlicensBilBdamBcomfiBblehBbanB8amB26BwafflBvaBresistBlegalBbilliBprincB
particularBlenoBdemoBconstantBchristmaB2010BtubeBsourcBshedBnovembBfestBericBdaisiBwoahBsniffBpiercBknightBjetBdareBcricketBtypicBstaffBremainBpartnerBnightiBhockeyBfloodBbedroomBawaitB	afterwardBzoneBtheatrBleakBgraceB	apprenticBamandaBÂ¹BsoonerBentriBblindBhahahahBexplorBeuBdrumBcentralBwindiBplanetBmitchelBlionBkobeBindianBwarpBsoftbalBpasswordBjstBinsurBdiBawwwwwByahooBtidiBswingBscotlandBpocketBmelbournBadultB2amBunfairBdisgustBcutiBchilliBspareBroofBhaterBeventuBdamagBconcernBreunionBheadphonBdrainBrtBitchiBallenBadvancBspokeBcombinBbrianBblushB
accomplishBmusicmondayBflickrBcmonBbeinBawkwardBasianBtempBtastiBstrikeBpopcornBdeliveriBcomediBbuzzBauBlogoBlemonBindoorBapptBwiseBwalletBspillBrangBleeBhoorayBfyiB6happiBtechnicB
sweetheartBshoutoutBrejectBpoisonBoweBlayoutBlapBhonBhikeBgoldenBgmailBtueBsoftBpeeBmmmmmBluckiliBhasntBhanginBvampirBseBoperBlangB
disneylandBaudioByayiBsheetBrachelBgeezBfailurBcrawlBbastardBamazonBttBtheoriBsausagBÂBretardBmedicinBemergBdiegoBtgifBsemestBgonaBcomplicBÂBtbhBsomethinBsmithBjakeBgudBgeniusBdivorcBbridgBbiologBsyncBparanoidBlvattBhavntBchoreBbellaBstolenBsightBrescuBobamaBmahBcowBbrowserBblipBmaintenBgniteBdemandBahhhhhBwaitinBtrynaBtoiletBmummiBgreekB5happiBvirtualBshouldntBmapB	wordpressBtwistBmeeeBenviBconvoBÂ¶BsunglassBselectBrelievBplainBbelongBayeBÂBsoloBscrubBpanicBpandaBlisaBgrandparBdrewB
dissapointBdickBaussiB21stBsubscribBsolvBplaylistBmagB8thBvodkaBsevenBsayinBremixBphotographiBbeforBpolicBphewBnegatBmeganBproducBpassionBkyleBjenBelectB
courseworkBwilliamBtakinBmowBmaBlauraBflBÂBsobBsliceBbegByouuuBwirelessBwhoreBturkeyBrainbowBÂ§BvibeBohhhhBnicolBjoeyBhvBengagBconvertBconditBaccentBwthBfoBbrushBrockinBprizeBjimBfrozenB400BzeroBtorturBteeBsmoothBpublishBmichellBfarmBemmaBdunB18thBolivByouthBwristBtmrBstrangerB	highlightBÂ·ByerBrestorBencouragBconcentrBpapaBlemmBirritBdenverBcurriBrecitBpiratBewwBbrideBphilBcoworkBcartoonBbakByoungerBuselessBullBtrashBspeakerBmarchBÂ­BwireBtigerBstairBshadeBpornBmetalBmaleBlowerBlebronBdebatBdanielBactorBwkBusernamBunBsubBshiniBscottBroleBpathetBnbaBnadalBmedicBheapBgrateBgooBproveBinstantBdedicBbudgetBboomBbackupBtaskBownerB	nevermindBjazzBgalBtwitterlandBoperaBniggaB9amBthousandBohioBmosquitoBloopBliliBhappierBgigglBgaBfortunBdumpBdangerBcommunicBwrkBwkndBtransferBtitanBteBexchangBemoBdustBauntiByayyyBslipBhomemadBcoolerB
bestfriendBtxBremotBnervBexplodBcarpetBboundBblisterBstatBmangoBflowBdullBattractBsolutBskinniBreaderBnuggetBhorrorBgrindBedgBdislikB27BworeBsinusBrobinBreactionBmonitorBcareerBarmiBunpackBprepBnorwayBhollaBeasterBadoptBwhewBthaiBguidBeminemBannaBwreckBpupBminusBdopeBdessertB	christianBbuenoBÃ°ÂµÃ°BsunbathBslideBpropBpotentiBnkotbBkayBeraB	downstairBassholBvoluntBstephBsocietiBsleepovBpoemBdeadlinBbassBveggiBteenagBfederBtendBsoniBholB4happiBtooooBshatterBpukeBpredictBpreciousBplateBoooohBforecastBellenBchemBcaffeinBwahBtwitterberriBsignalBrawBpleasantBpeteBmurderBiÃ¯BgoodiBgeeBexitBantBÂBÂBwudBtwittBttylBsecBsantaBpurposBpinBkiddoBjamiBhiyaBeatinBconfidBclaimBbagelBwhooBsuckiBsoooooooBlegendBjacobB	earthquakBblanketBadvertisBvineBtheydBtheeBsimonBrobotBpatrickBequalBclosetBparamorBkristenBieBfawcettBunablBpregnantBpileBpepperBneBnawBmysteriBcoolestBwipeBlaidBjunkBhunniBduhBdomainBannB7happiBvictoriaBpraisBglasgowB	geographiBoppositBjohnniBdevilBahahBvistaBthrillBroommatBnxtBjoneBflickBabB34BwhoeverBumbrellaB	statementBsonicBsoberBpenguinBneglectBitchBindiBfootiBbalancBsilverBproposBpitBovenBloBpalBgrumpiB	argentinaBaaronBthomaBsafariBroyalBinchBfavorBdonniBblissBoutdoorBmarioBbraceletB28BÃ°ÂµBrogerBretirBreckonBpolishBlecturBhintBhaloBdiaBaffectBÃ°ÂµÃ±BzBsangBjessicaBgoooodBdublinBdisplayBcastlBboilBsuppliBsatisfiBnomBnjBmarleyBhoodBeffBdyeBcentrBanxiousBuberB
straightenBregardBmintBlinuxBhabitB
soundtrackBsomewhatBlisteninBianBhigherBeveBultimBtherapiBthatdBspymastBmp3BboredomBbatBsnlBroutBratBpixBfoneBblewBbecamB32BurgBtehBstretchBsaltBproofBpromoBhardcorBdelightBafricaB2nightBwhaleBuponBtweetupBtapBstephenBshouldaBphoenixBincreasBhedBdohBchainBcampusBtrapBstackBjuniorBgeneratBframeBbashBunlikBtallBsortaBroveBpatchBncBmidBgrantBfundBfrigginBflewBcirclBassistBvirginBthemBstunBshudBsandBribBreceptBrabbitBintroducBhpBdinBcruelB	countdownBachievB95B8happiBstandardBsaraBroutinBinboxB
hahahahahaBeffinBdorkBaidBxxxxxBubuntuBseanBpsychBportB	overnightBmirrorBleaguBjenniBjacksonBexamplBeducBawarBalternB7thB2000BjournalBgurlBgmBfrankBeaBdisturbB10amBvaluBstringBrealiBozBnineB	clevelandBchopBasot400BxoxBwhileBvancouvBstalkerBpursBpadBlbBdvrBdizziBcleverBauthorBsolidBooooBhypeBbrosBakoBÃ°ÂºÃ°BusbBtowerBthreadBsoxBkarmaBextendBdrownBbsbBÃ¤BvalleyBunhappiBrichardBmultiplBleavinBinjuriBdistancBcoatBbouncBbondBbloggerBbiscuitBbeyoncByogurtBworkshopBunexpectBtickBplurkBikeaBbenefitByuckiBspecifBrewardBpitchBperriBnecklacBjerriBitÃ¢BfantasiB3happiBtubBsubmitBsrsliBsakeBpoB
photoshootBirishBdougiBdetroitB	cheesecakBboobBbeefBÂBwifeyBtwittervillBsuspendBsilencBpatternBkoreanBhahaaBgirliBdiscBcornBcanadianBapBwonBsaddenBprisonBplasticBbonusBbackyardBamenB
supernaturBsmoothiB	pattinsonBntBnovelBkitBizBgumBchampionBadvantagB6pmB15thBÂ¤BskirtBmateriBmanilaBgangBformatBcoveragBcomboBcharlottBalicBwhoseBturtlB
tournamentBshaunBsdBnaughtiBjrBdiscountBdevicBdescribBnhBmegaBchewBbeastBashamBvanillaBtwasBsodaBpostponBjointBheadinBgingerBditchBberriBaaahB2happiBwebcamBneatBmeeeeB	childhoodB1amBÂBtatBsuprisBpeBmenuBheldBfurniturBcharitiB	butterfliBazBabilB5pmBÂBsipBrioB
professionBmaskBknwBjeezBhencBgatherBchiBbrbBbradBbasementBabandonB99B10pmBvenuBsofaBsamplBprBmatthewBinventBflavorBconceptBcocktailBbraveBanatomiBvomitBtheirBseparBptBoclockBnashvillBknitBhundrBgoodnitBgiveawayBfactoriBdiffBbeatlBabusB9happiBwornBwithdrawB
washingtonBtragicBsunsetBrliBrecoveriBnotebookBmiraclBhottBfeeBculturBcreepBbinBÂ¢BÂBunbelievBtadBswedenBslapBsalonBrapBnzBnooooooBneedlBlizBhalfwayBdocumentBchromeB	broadcastBbotBbizBspammerBpatBmugBmandiBlattBkatBhandsomBbullshitBaahBwkendBupsidBsurroundBspaBquarterBoveralBlukeBjelliBheyyyBgateBewwwBdecembBblankB
australianB31BwendiBportlandBhumorBcommitBberlinB	amsterdamBthouBswallowBo2BnicerBlosBhardestBgrewBdhBcooperBchefBchampionshipB17thButterBtaughtBpeachB
heartbreakB
friendshipBdisastBdenBcrystalBchiliBboooooBaiBgrayBglobalBeurovisBdevB19thB16thBupcomBtumblrBlenB
internshipBiamBfallenBcageBboothBbiblB	wimbledonBthusBthinBsquirrelBsinkBshBscanBpeelBpatiencBpassportBparticipBiremembBflopBbobbiByankeBstripBstoodBmessiBmarriagBjoBjavaBhorniBfarmerBcheaperBbowBalienByeBstickerBsnifflBroomiBplayoffBnokiaBnintendoBmelBfloatBarguB75BtornadoBtentBpunkBminorBjailBcalendarBbuiltBballoonBagentB182BthereforBpluginBimmediBfilterBcountiBconsumBchoseB	autographBxmenBswapBsprintBpeaBmcBhubBdipBborderBbaB	tommorrowBkaBhughBhecticBgoodluckBftBcribBarrangB	archuletaBwotBworshipBsoapBrecognBramBfamiliarBdeniBctBchileBbrookBapiB10happiBughhhhBsuspectBrerunBneighborhoodBmarB	houseworkBhousewBgooooodBgimmBdancerBcostumBblownBaveragB150BzacBtossBtaxiBsewBreservBnumbB	margaritaBjawBinkBeekBdrinkinBcrimeBcheekBceremoniBbrowBaghB13thB0pmBuhhBtabBsparkBpresidBheatherBcontemplBcaloriBbrickByaaayB	volleybalBrecessBrantBmartinBljBlambertBlagB	grandmothBfadeBerrBeconomiBcrackerBbikiniBadmirB9pmByearbookBstrengthBsnoreBriskBqueueBqualifiBonionBlungBhoodiBfaBetsiBddB3pmB10thBupstairBtomozBpearlBmightiBlegitBg1BeightBbreezBaliB9thBÂ¯BsilentBluciBlovatoBinjurBhiccupBggBdutchBdeanBcigarettB2pmB29B20thBtrainerBswellBsourBshiBpanBlobsterB	liverpoolBgeorgiaBexpirBdeterminBcbaBcaptainBaveBautoBattachBamberBacoustBtetriBsytycdBrumorBpokeBhiltonBdraftBdesertBconfessB8pBrenewBoffendBmyweakBmichiganBhottiBfrogBdiveBbrandonBbonBouchiBlushBkaraokBjaneBhailBgariBfoldBdodgerBcrabBcontainBcapeBÂ¨BwierdBrossBreinstalBorientBmolliBmariaBexpertBcarolinaBbonfirBbestestBbeardBbamboozlBabcBwutBwhipBurbanBtylerB	transportBtomoroBsaddestBreliefBninjaBmelissaBgcseBforestB	edinburghBeclipsBderbiBcubBapproachBalliBtypoBspoonBrestartBlayinBfunniestBelbowBdiseasBbidBattitudBsprainBsouthernBsetupBresidBrafaBopBnwBmuseBmicBitouchBhoeBextensBeuropeanBchocBburritoBboozBalanBterrifiB	summertimBsteamBsprayBromantBresetBparadisBlemonadBintentBhuluBgarlicBdotBcudBcrankiBcoldplayBbelovBbalconiBadditBvillagBprogrammBperthBmakerBlightenBknowledgBindustriBgregBdoseBbrooklynBwooooBverizonBtoneBsackBrugbiBporkBmontrealBdellBbushBbrewB420BhazBfmBfairiBcoloradoBbritBautomatBwhtBwaleB	uncomfortBtelliBspotifiBovercastBnamanBjareBickiBhandiB	experiencBdelBuhmB
twitterfonBtbBshaneBowlBoprahBnewcastlBloginBleftovBleaderBfellaBcsiB7pmBzealandBunemployBtoothachBspearBporchBoreoBnikkiBlololBlitBlambBknackerBjackiBhammerBgonBgnaBcelticBcarrotBwidBsonniBrickBnetflixBintendBhappiliBgtBdooBdestinBconsistBchokeBcheesiBcampaignB	brazilianBanxietiB48B101BÂBtequilaBsupperBregentBrbBmikeyBhamBdragonBdiamondBchinBalertB1happiBtodoBthrownBsemiBresumBrankBradBpspBlickBhomesickBdineBdenniBdecorBclosestBbradiBvictoriBtuBprinterBpixarBnewestBmanualBmaamBjonathanBhyperBheartbrokenB	constructBbirthBbelivBbelatBahahahaB2008BwwdcBtoooooBtmobilBtampaBtabletBrageBlooovBinitiBfireworkBfarewelBexpoBdoughnutBburstB4pmBwhisperBtutoriBtrialBtelevisBsurvivorB	sooooooooBsmackBrubiBnpBhuhuBharshBfunnBdismayBdefaultBcouldaBcarradinBbeltB700BstitchBspiciBsnakeBseekBsammiBoutlookBmournBmannBlabelBjeremiBh1n1BfontBcumBcraftBcandlBayBanticipB600BÂByummByouuuuBtowelBsuicidBspencerBshadowBsalvatB	overwhelmBgovernBemployeBdevastBconsolBcondolBatlantBactressBabbiB	treatmentBthesiBsweatiBstrokeBnighterBnestB	necessariB	motorcyclBinnocBihopBgmornBwoundBvisualBtornBstuffiBsinBsentencBreflectB	pointlessBplotBpenniBorganisBlouiBleonBlaneBermBdroolBcharBackB530BsupBseedBromeBpbBpathBkeithBhboBgrapeBexhibitBdominBcurvBbnpBvitaminBsympathiBsophiBreplayBnotifBmodernB	jailbreakBfredBetBestBcodBappropriBabitBwhineBthirstiBthickB	spongebobBslackBmtBmissiBmaiBkoreaBgranniB	franciscoBfinanciBelliBdrunkenBdiskBdiscoBcrispBcoreBcolleaguBchillenBbitterBbettiBaugBarizonaBarchiBwilBtescoBstickiBresortBquestBpunishBpjBpineapplBpassengB	paperworkBoralBokiBjoelB	indonesiaBemployBdepartBdaneBcrunchBbbiBbailBaudiencBargumentBÂBwackBunlockBuniquBtailBrockiBpintBocBnominBleanBlastfmBkkBinvestBhaiBdylanBdisablBdayiBbucketBassBstewartBsimpsonBseptBreligionBoiBmuchoBmegBlimeBleedBlarriBhaaBfranBdryerBcouponBchristBcheckinBboldBbeboBbarnB1pmBtissuBstilBspoilerBshufflBrocketBlunchtimBhollandBholaBficBexternB
disconnectBdemBcounterBcaramelBandreaB11happiBveronicaBtorusBpoleBpeakBpaulaBotBkillinBgrrrrrBgalleriBformerBeconomBdesirBconventBclassmatBccBburiB	blueberriBbettaByaayBwksBvocalBvictimBteddiBsleepinBslamBrusselBringtonBreducBniBkfcBhauntBegoBdonÃ¢BbethBasiaB85B55ByeahhhBtrailBtommiB	sunscreenBstevenBsqueezBpoppinBpermanBpastorBnatBlotteriBlewiBintoBhumBhometownBhmphBfortBbehavBahhhhhhBÂBÂBworkerBvisionBvirginiaBveganBuniformBtutorBtodayiBsponsorBshoB	oversleptBoocBinteractBgpBfunkiBfudgBcyaBchampagnBÂBwherevBwaxB
vegetarianBurselfBsweetestBreliBprotestBphraseBnkBmethodBloveeBloanBillegBheidiBfrankiBdittoB	contributBclairBbcuzBvisitorBvacayBrapeBoatmealBninBmultipliBlgBgahhBckBchartBbballBadobBwouldaBwolfBwanderBtwatBtuckBtriplBsymptomBsunrisBsmallerBscrollBnewspapBmickeyBlifetimBkeenBhowdBfartBehhBclinicBbootiBthangBshoreBrouterBrelayBoffensBmicrowavBmeaniBmarinBgawdBfuelBcraigBcommutBbrisbanBanthoniB42B100thBwifBtraceB
temperaturBsuckerBsrriBspB	portfolioBoldestBlighterBkmBhunterBhihiBgenBdrakeBdefeatBanniBanaBadjustBwaynBtwentiBshiteBrodeB	lastnightBjaBhappendBfuzziBacctB65BwallpapBvacaBstadiumB	sleeplessBslaveBscBposeBpoliciBphpBnetbookBmuahBleatherBjuriBintroBintegrBhotterBfrBfattiBfalsBconeBbrandiBbgBÂByoudBw00tBtrimBshownBsalsaBretailBpudB
pittsburghBmotionBmcmahonBlucaBisplayBhowardBgadgetBfrostBearringBdaviBbrightenBasidBwilliBspiBrangerBlooongBkcBhahhaBduBdoomBdebbiB2mozB25thB12thBwishinBwayyyB	twitterifBsleeepBsidekickBshrimpBrussianBrhymeBpausBmilkshakBkoolBiranianBimacBhorridBgwBgpsBglowBeyebrowBchampBcaveBcapitBadminB11thBwerentBswedishBstabBsissiBshinBrawrBmorganBkiddiBindianaBhehehehBfencBeddiBcopeBbraBballetBarrestBantibiotBalllBxoxoxBvintagBventBvanessaBtokyoBsheepBselfishBrollerBrevengBpimpBpaleBnataliBnanniBnanaBmÃ£BmwahBmessengBloooovBjoseBhkBfaintBfactorBelectronBdougBdouchBcalBboohooBbahahaBworthiBunlimitBscarBpreventBoleBnbcBmochaBlaughterBdoughBcarnivBbannerBandroidBairlinBshuckBscaleBrepBpilotBpepsiBno1BmonicaBevaBdyBdownsidBdebutBdairiBclownBchloeBcentBbuffaloBbatmanB	barcelonaBbachelorettBamongBajB11pmBvolumBtwitterworldBrunniB
regardlessBredoBpinkiBmodBmisterBmeasurBfameBdrillBdeprivBchoirBbrightonB
birminghamBauctionB90210B36BÃ¥ByoungestBpingBoasiBkidneyBhiddenBgunaBguardBgranBforeignBdunkinBdelishBcsBcanonBcanalBangerBsteelBsheeshBprimeBpokemonBomgoshBnoooooooBlyBfuBeuroBerinBcommandBbruceBbatchB30thBwormBuhhhBthanksBstavroBmaddiBjanBjadeBhveBgreasBfwdBendlessBdonBdlBdeeBcriticBcircusBcamperBbriefBvisaBureBstingBsharkBrereadBraveBplatformBpawBpatioBnÃ£BluggagBlogicBistBinfluencBhangoutBgoddamnBgoatBfourthBfollowinBenemiBdisagreBcourtneyBbrutalBbadassBannualB12happiBtonsilBsyrupBrashBperezBpacifBovumBneilBneedaBmethinkBmdBimpatiBhutBflakeBdiariBdecBcellphonBcasaBcanÃ¢B	broadbandBampBwoeBwizardBtylenolBtreyBsweaterBstrongerBstrepBpaymentB
millionairB
membershipBmaggiBkudoBjusticBjuliaBitÃ¯BhighwayBheyaBgueBferriBdirtBchicaBbrakeBbaileyBadvertB2mrwBtreadmilBtragediBtinkBtherBtedBspokenBsharpBsgBsailBrebootBpreordBpollBpeekBowwwBnathanBmunchBmimiBkeBjohnsonBjamminBinfamBincomBgbBfolderBcuppaBcookoutBcinnamonBagendaBadaptB64B630B11amBtreasurBtacklBswayBstereoBsodBslowerBpjsBpinchBobjectBnevaBmooBleoBlappiBhenriBheightBgreecBfootagBfishiBfilipinoBexclusBdirectorBcvBcenturiBcalculByousByaaBwoooooBwesternBwassupBunionB	ubertwittBtskBtraditBtollBrentalBpineBpacketBoutletBkenniBintelligBhottestB
highschoolBhamsterBestatBelephBbunBbreastBbomBblogtvB250BweighBteeheBsuitcasB	reschedulBraidBpanelBmerciBmarvelBlearntBgloriBfoundatBflirtBcordBcomplexBboBbleachBarghhBallllBwhackBrunnerBpowBpongBmockBlahBinlawBhurricanBevidBdefoBcrisiBcowboyBcodiBcategoriBcasinoBbarkBantonioBahahahB5kB3000BÂBunusuBspectacularBsalmonBralliBporBpaceBnvmBnonstopBmeowBlikeyBkickinBjjBgotchaBcolumnBcocoBcleanerBchipotlBchaBbulliBbrittaniBbegunB900B247BzachBupperB
twitterfoxBtransitBtetherBstainBspiceBspecBrecievBoregonBnaviBmusicianBmmmmmmBmillBmidtermBloooongBladBkrispiBkaylaBinaperfectworldBhungerBhomelessBheaterBgarbagBfreezerBflawlessBevanBdmbBcrownBbookmarkBassemblBashtonBappealBxpBtunaBstoragBsketchBseminarBregionBpunBponiBpetitBovertimBnorthernBnascarBmadisonBkongBknowwBincidBhtmlB	ghostbustBfurBfringBcockBÂB
understoodBtorrentB
throughoutBstayinBsnoozBrockstarBrickiBrepresBnytBnowadayBnateBinjectBheeheBfreakiBfiancBdollhousBderBcubeBcramBawwwwwwBalphaBaaB447B333B08BzomgBwembleyBunderB	trampolinBtoddBsueBstiffBshoppinBseoBsandalBrevealBphaseBmalaysiaBlmaooBjussBjunBjobroBingBindexBgloveBfinancBchaiBassassinBargBallisonBwardrobBtimelinBtcotBsurveyBshouldvBshannonBsarcasmBrumBromancBrecalB	psychologB	pricelessBpoetriBoverloadBlykBkingdomBinterfacBhappiestBfateBeaglBdenmarkBcreekBcontBcapturBboostBbbmBairplanB29thB00BwiBudBtoddlerBtivoBtingBthoroughBtaleBstrandBskiBrestlessBramenBmoiBmemorBhumpB	complaintBbeckiBbeccaBalyssaBaloudBzoeBytByayyyiBtnBthtsBthanBsickiBshorterBloolBlollBlcBhollyoakBharperB	grandfathBfobBeditorBcompetBcaseyBÃ¦BwrestlBwhatchaB
watermelonBunknownBtowBtinaBtiffaniBsowwiBoccasionBnikeBmushroomBlimpBjanuariB	interruptBinputBhiiB	haveyouevBgmaBforkBflameBffsBespressoBdubaiBdormBcullenBcouncilBbuffiBbuffetBadvisBadaB800B22ndBxoxoxoBvegBsummitBstickamBsplashBsalliBrecruitBprioritiBpleaseeBperoBoverrBnoahBnickiBmemphiBleaBkidnapBjunglBjeBimaxBhobbiBgentlBfascinBempirBegyptBdocumentariB	discoveriBdinerBdashBcoziBcmBceilBbuseB	breakdownBassociBaleBagencBtraviBtomarrowBthighBshuttlBrecBperkBoscarBnmBnicestBnewbiBjessiBinsultBindividuBhumblBhamburgBfuckerBelevBdeerBdaniellBcorruptBcompatBchaoBalikBakuB27thBzackBvalidBvacuumBuiBtaraB	spaghettiBsexualBseeinBrenderBpuzzlBpropertiBpicklBoutingBoptimistBmiteBlendB
kardashianBkansaBindependBhayleyBguaranteBdreamtBdefensBcÃ£BconflictBconclusBcoincidB110BspreeBpresencBpondBpedicurBnudgBnileyBnauseousBnadaBnachoBleopardBledBlautnerBisraelBinstructBinsistBhawkBgrammarBformalBflippinBfictionBdarlinBcomcastBcnnBbenjaminBbadgB26thB105BÂ®BthailandBtayBstripeBsmelliBslumdogBronBredbulBmildBliBknifeBiviB	implementBidealBheeyBfighterBenviousBdewBcolumbusBcarbBbirdiB24thBzzzByurBydayBtartBswagBstatistBsprinklBsoyBsleevBscoutBsacBresourcBpediBpaypalBpattiBoccurBnovBnicknamBnearbiBmustvBmjBkarenBjewelriBhittinBhasslBgenuinBgaspBfreshmanBeasternBdukeBdaniBcarlBbyebyBbillionBbasketBalaskaBwalkinBrailBperuBpandoraBnuBnonsensBmixtapBmatBleahBlayerBlatinBinvisB	hairdressBghettoBdodgiBcropB	christinaBbelgiumBbackwardBacademiB2007B10kBventurBvariousBuppBtomoBstreakBstampBsbBnoisiBnewslettB	multitaskBmudBmentBlivinBklBkewlBitzBhmmmmmBforthBequipBdivaBcorporBcoleBcharlBcassiBcarloBbristolBbamBarghhhBaightBahemB23rdByummmBwarmerBwaaaayBspitBremediBomBmamBkeeperBjennifBintriguBillustrBhashtagBgiBgemBdatabasBchapBcestBceptBartworkB72B44B430BtpBterriBreallliBranchBpodBpencilBowwBmovinBmiloBlaxBjarBinnerBikrBickBh8BfluffiBearlBdescriptBcobraBciderBbaltimorBaffairByaaaayBsooonBscottishBsarcastBremakBpaydayBpatronBorleanBoooooBobservBnetbalBmoanBmechanBmariahBmadridBlotsaBjulianBjtBhurrahBhardwarBgoooBfordBfacialBdrivinBdexterBdecadBcallinBbyeeBbahamaBasthmaB10000ByukBwntBwhiskeyBuvBurgentBtotBtnxBshldBscamBrosiBrichmondBpaycheckBpamperBninaBngaBmashBlesbianBkremeBinvestigBhopelessBgitBgapBferBcriminBcoasterBchadBtinBsaintBmussoB
masterchefBlolliBkathiBimoBhowdiBgleeBgeneBfrequentBfreelancBfogBfiercBdelhiBcrankBcondoBclassiBbarriBtwitchBtriviaBtonguetiedgtBtmwBstrategiBscarfBricoBrevolutBperfumBoklahomaBmotorBmbBinsightBinsertBheadlinBglastoBfrozeBfirmBeliminBcrowBcoconutBcartBbooooooB730BymBwivBtweetinBtweakBtjBrecyclBpollenBpfftBpermissBpauloBpaneraBoakBmumbaiBmellowBlÃ¯BliquidBkristinBiloveyouBictBichBhamiltonBglueBfuriousBfirefliBfinlandBevertonBdrummerBdockBdinosaurBddayBcreaturBconsultBcollapsBcachBbrighterBathletBasdaBarenaB46B41BtunnelBthiBtehranBsumthinBsoothBslutBsistaBseafoodBretreatBooopBokayiBofcoursB
netherlandBlindaBjayzB
instrumentBingrediBhomepagBheeeyBgoooooodBfistBfabricBespnBelementBeachothBcolinBbulletBbryanBbblBbacBamazinB2hrsBwelBwarriorBvickiB	underwearBufcBtraciBtiffBshampooBscooterBsafetiBrottenBrcBmÃ¯BmountBmeeeeeBmaidBlousiBfallonBembarassBbaldB4gotB38B14thBwiffBweeeBunluckiBtonguetiedquotBthrobBtemplBstormiBsocalBslimBsiblBscrapeBsandiBsadfacBrewatchBrepostBpgBouBnuthinBmovementBmodemBlegoBkiwiBkindlBjuzBjosephBjdBirlBhurrayBgokeyBgilmorBflyerBfingBfiestaBembracBeeekBeagerBdwBdubBclashBcertifBcabinBbrunoBbrillBappetitBalthoB97B39BÂByeaaBwheredBunavailBtropicBtracklBstephaniBslotBshelterBraininBprofBpermitBoutagBomjBmartiniBmarcBmannnBlotionBliarBlengthBlaborBl8rBkenBiplayBfrickinBformulaBexceedBenvironBdramatBdiaperB
commentariBcoffBcarolBbaliBanyhowB830B07BwpBvivaBverBtuitionBtipsiBsupermanBsulkBstartinBskittlBsincerBsensitBroadtripBriotBpcdBohhhhhBnodBmeetupBmbpBhoopBgotoBgahhhBelviBderekBdeployBcottonBcoopBclapBboreddBblahhBbittersweetBbackstagB98BvegetBspockBsoldierBrdBproteinB	professorBoutcomBnonethelessBmodulB	manhattanBloovBlonesomBliquorBlifestylBlalaBkyBkeynotBgovtBfosterBforeheadBdintBconnorBcommencBcloneBchosenBcardioBboxerBboiBathenB43B130B125B120BwheatBuhohBstablBregistrBpumpkinBprimariBpaigBoxfordBooooohBmurrayBmamiBlolaBlashBkungBjuiciBjennBicarBhonourBheathBgroveBgravitiBdozenBdiggBdeffBdaysBcostaBcompilBcatchiBcardiffBbranchBbeadBangiBaloB2getherB28thByungByippeBxmasBwaaayBviolencB
unfortunatBsposeBshawnBropeBreversBresolvBrapidBquickerBpussiBpuertoBnemoBmrazBlindsayBjodiBjillBiowaBimiBhogBhighestBgeoB	gentlemanBeternBenablBdeclarBcindiB
cheeseburgBbroadwayBbookstorBblendBbicyclBbendBbelfastB69B1030BzipBwldBweaponBwarrantiBviolinBtysonBtourneyBstanleyBspikeBshelbiBsequelBscrapBrubberBpowderBpamBmoniBmangaBlvBlampBjumperBincBgabeBfunkBfulfilBfirmwarBfedexBexpandBehhhBdosentBdmvBdefendBconservBciaraBchillaxBcapablBbuggiBbridalBboltBblokeBbeatenBbawlBbarbiBavBalgebraB30minBwitchBwidgetBwayiBwaistBulcerBtitBtamBsayangBrumourBrestrictBrelaxinBpuffBontdBodBmyselfBmayerBlineupBlawyerBlandscapBimpactBgraveBgagBfreebiBfluidBfinnaBfewBexplanB	elizabethBearphonB
difficultiBdiddiBdiabetBcouragBcottagBcoinBchemicBcedarBbloomBbecuzBarchivB96BzuneBwhistlBvanishButahBtilaBthreatenBtechniquBsÃ£BstudBsparklBshredB	sentimentBscramblBsagaBprixBpartialBoliviaBoccasBnicBmuBmnBguiltBgloomBglobeBftlBfleaBencountBdwightBdongBdialBciaoBcasualBblakeBawwhBalllllBtcBstrollBstinkiBsaoBrackBpythonBpugBpreorderBportugBpleaseB	nooooooooBncisBnativBmeantimBlizziBkirkBimmensBidentBhornBgoodsexBgeekiBfountainBfondBfeastBeepBdominoBdepositBdebtBdangitBcrunchiBbuilderBbanquetBalrightiB56ByorkshirBwuzBugghBthnkBtaiBsozB
soooooooooBsamanthaBpipeBpajamaBottawaBoppBnostalgBnicholaBmaturBlabourBiraqBhadntBgomezBdripBdpBdentBcupboardBcreationBcomebackBcmtBcluelessBcivilBbewarB	badmintonBampwinkBafricanBaaaahB1111BxxxxxxBweekdayB	voicemailBunwelBtolerBthatllB	temporariB	subscriptBsquashBsethBscifiBsavannahBquizzBpuffiB	portuguesBpimplBperspectBpainkilBnanciBlensBlasagnaBknotBjinxBjelousBjakartaBinfrontBhtcBhindiBhelenBgreediBelsewherBdistrictBcurtainBcourtesiBcooolBchucklBblurriBbingoBawakenBalleyBaidenBadelaidB92B78B100000BvinylBvehiclButBuggBtouristBtoteBsuspensBsucceedBseperBpixiBpiggiB
philosophiBpeniBojBn97BmonopoliBmilwaukeBmereBmelodiBmadrBliverBipBhugglBhowrBhookahB	heartburnBgownBethanBdriftBdentalB	classroomBchristinBbckBarntBalabamaBajaBaidanBadvilB
wonderlandBwikiB
underneathBundBummmmBtrentBtooooooBtisdalBthirtiBtemporariliBtakBsweepBsorrowB
skateboardBregBrearrangBprospectB
playgroundBpartayBparBoldiBnatalBnaoBmutualBmeanwhilBmanniBlincolnBlikBknowwwBjasminBhumourBhsmBgtgBgrandadBgotaBgiddiBflavourBdrivenBdestiniBdayyyBcostcoBconfBbiBattBangelaBalexiBacidB4everB2bB102BwilsonB	wikipediaBtwhirlBtributBterrifB	substitutBstrapBstaplBstallBsleeeepBsaneBrewritBrefundBrearBrachBqldBpresalBpractisBponderBnowwBnoblBnauseaBminimBmermaidBmelbBmaintainBlaserBkasiBjambaBincaBhuhuhuBhoooBherbBhashBgrowlBgraderBgalaxiBflatleyBfeatBduperB	disapointBddubBcÃ¯BblurBbluntBbeaBautumnBanalysiBÃ¡ByarnBwittiBwhoooBvibratBstumblBspankBsoggiBshhhBromeoBrepublBreminiscBprototypBportablBoyBomggBnvrBmorrisonBmiseriBlitterBizziBironiBinspectBinnB	householdBhawtBhardiBgrahamBfifthBfavourBenufBenjoyinBdonkeyBclarksonBclarifiBcarterBbenchBadrianB230ByangBwanBvlogBviBvh1BversBtutBtrueliBtemptatBsyndromBsymbolBsubmissBsteviBspeltBsatellitBsailorBsaddBrunninBrotatBretainBrebelBpigeonBouttBobvBnÃ¯BnewliBmargaretBmaddBloveeeBlooooovBhushBhiiiB	hahahahahBgroovBgordonBfinBfagBessentiB
dictionariBdetectBdeepliBdearestBdeafBcommBcollarB	cheerleadBbuBboaBblairBavenuBachiBaaaB5000B1500ByeapBwhyyiBustreamBughhhhhBtwiggaBteaserBswissBstungBshelfBseshBrandiBradarBpuppetBoverseaBoverduBnowwwBmurphiBmpBmothBluxuriBliamBkettlBkerriBkentuckiBjustifiBindulgBhippiBhahahahahahaBftskBfreddiBdevonBdehBcpuBcocoaBcigBbuhB	atmospherBamoBahhaB2006BÃ®BÂByeayBwhydBwelllBvelvetBunrealBunlovBtofuBtmBtennesseBtemplatBt20BshoveBscoopBrustiBromBriderBpronouncBpottiBpoloBpilatBparaBothBmoodiBmonsoonBmojoBmobBmcdBmarshalBlÃ£BlonerBliberBlibBlettucBlanBlaceBketchupBinterwebBinevitBhysterBhousemBhelplessBhddBgenerousBgamerBfunnnBfracturBfeckBewwwwBdowntimBdeffoBdaylightBdammB	copyrightBcongestBcameronBbubbaBbargainBaudreyBanywhoBanglBahhhhhhhBagoniB930B62B52BzoomBygByestBtruBtrophiBtrippinBtomorroBstrawBshortenBsacrificBrobbiBrehabBportraitBoptBontarioBmarcusBlagiBkanyBjennaBhhrsBhaftaBgummiBflushBfebBfcBebookBdomestBdbBdarkerBcontextBcommissBcircumstBbradleyBbffsBafteralBaffiliBabiB67B58B1200BzeBweirdoBvbsBtÃ£BtweoplBtlcBthankiBtellinBtdBswitzerlandBstrictBsneakiBsharonBschooolBremarkBreliablBracistBposhBphishBmuhBmistakenB	metallicaBmentorBloriBloganBjeepBhorizonBheyyyiB
elementariBefronBeeeBdunkBdownerBdetoxBcoronaBconveniBchuckmemondayBbratBbeganBbarrelBbakerB115ByouuuuuByknowByankBwweBvouBtrollBtimmiBsupermarketBsundaBsomBshowcasBpostcardBpixelB	paparazzoBpalacBnearestBminimumBmilitariBmeditBmcrBlouBlikewisBleafBjammiBirvinBiplBiniBichatBhongBhoedownBhidBfroBfowardBfeatherBexplosBerrrBdrupalBdianaBdanaBceBbiasBbackpackBancientB	accompaniB82B54B49B47B311ByayaBvenicBuuBtyreB
twitterbugBtrickiBtileBsurpisBsolarBsmoreBshiverBrugBrnBrlliBreunitBrebeccaBplazaBpaddlBomgggBoctBmoorBmkBmigratBmarvinBmagnetBlobbiBlmaoooBliteBkiBjpBjordinBindicBhotdogBhankBgriefBglitchBgetawayBfussBfotoBflatterBexcessBelliotBefficiBduetBdonnaBdolphinB	conditionBcnBcalvinBbladeBbizarrBbeleivBandersonBaddiB89BÃ°ÂºÃ±BÃ°ÂºByouÃ¢BwiveBwazBwalkerB	timberlakBthnBswampBstirBstartupB
speechlessB	sheffieldBseesmicBrussiaBrockbandB	quarantinBnerdiBmoralBmoderBmauB
leadershipBjetlagBjanetBjackmanBiiiBhurtinBhissBhatinBhatchBharveyB	halloweenBgrumblBgavinBfoggiBericaBdrenchBdontyouhBdingBdegrassiBcuntBchufB	chillaxinBcambridgBbeliefBarabBandyhurleydayBamongstB88B51Bxboxe3BwoohB	withdrawlBwhyyBwesBvodafonBturkishBtriggerBtrancBtksBsuBstompBsouljaBskintBsitterBsermonBsellerB	scrapbookBscottiBsashaBsambergBsadderBreesBprofitBprecisBprankB
powerpointBpeedBobnoxiBnurseriBnihBneonBmilanBmarylandBmakeovBlolololBlogiBlilliBkiteBkartBittBinningBikBheaderBhaulBguildBgoldfishBfinaliBenuffBdrawerBdonÃ¯BdigginBdiceBdaycarBcoriBcitizenBchunkBcanuckBcamdenBbfastBbarbequBanthemBaccBabsentB911B57B4wardB37ByeaahBwompB	wisconsinBvicBunplugBultraBtoreB	throwdownBthreatBterrorBtechnoBstaceyBsomthB
silverstonBsignificBsignaturBshiaBps2BpremiumBphoBpearBowenBoohhBmopBmansionBlatterBkevBirBintBin2BhopinB	honeymoonBguyzBfoughtBfinnishBfiBevriBenrolBengBellBdrawnBdivinBdimBdigestBdeptBcryinBcrispiB
crackberriBcorpBcoreyBconquerBcarolinBcaitlinB	bluetoothBblazeBbeamBbakeriBariBalbertBÃ¬ByewByaaaBwhyyyyBweepBwagonB
tweetheartBthnksBspurBsnickerBsneakerBslashB
screenshotBrumblBroxiBretroBrebuildBpoopiB	placementBp90xBnewportBmattiBkeriBkelseyBhaikuBgripBfundraisBfrightenBfangirlBetonguBeliBdtBdepotBdeclinBcuterBclayBcareyBbragBbollockBbnBblurayBaustriaBallnightB68B103BzzzzByasminaBwigBwelshBuptoBundergroundBtrBstruckBstaciBsickkBrevBpolarBosxBnyquilBnswBnagBmirandaBmiiBmiddayBlexiBjunkiBhoseBhooverBgtaBgovBgamblBfuckenBfoulBferrariBfalloutBexecutBelevenBdwnBdownhilBdiyBcreeperB
craigslistB	clothdiapB	christophBchesterBcheerioBcalebBboredddBbangalorBÃ§BwasherBwaaahBunsurBtommBtierdBticklBswordBswimsuitBsurfacB	superstarBstellaBspanBslumberBshadiBsanaBritaBrascalBpuddlB	preschoolBpregnancBposseBpierBpiBparallelBnovaBmojitoBmendBmattressBmafiaBloyalBlauriBjsutBjasperB	insomniacBhomeeBhelmetBharmBhaleyBgrubBgooooBfuseBfareBexaminBdrivewayBcpBcookinBclimatBblackoutBbeveragBbackkBbabysittBautomBaquariumBapaBalohaBaccurB20minB15minB106ByipeBvaccinBusagBtopshopBtireddBspontanBspirituBsmokinBschemeBpromptBprobliBpriorBpoohBpickiBphxBoughtBomggggBmanicurBmammaBkeepinBjulietBjensenBjealousiBitvBinvadBheaviliBharBguineaBgrrrrrrBgraviBgraciaBfrickBfreindBelderBdÃ£BdumbassBduckiBdrizzlBdoooBdkBdissertB
comparisonBcolognB
cinderellaBcaraBcalgariBbuatBblizzardBblipfmBbiggiBbeijBbebeBbattlefieldBbatterBbangkokBalmondBabroadBÃ¨ByeeBxrayBwhatsoevBviceBtwugBtwiiterBtokioBteariBtalagaBsuitablBstoopidBsleeeeepBretrievBreligiBrelevBquiltBpussycatBpoppiBpolandBpaintbalBoptimBobligBnedaBmoeBmauiBlockerBlalalaBjqueriBjolliBinhalBimmunBhideousBhenBhahahhaBginB
friendsterBfriendfeBfreewayBfallinBechoBdreariBdodgBdiagnosBdebitBdatelinBcrutchBconsiderB	coachellaBclarkBchessBcavitiBbyeeeBbreakupBboobooBbandaidBawhhB79B1100B104ByanBvettelBunoBuncoolBudahBspinachBspeediBsoarBscrabblBreneBraBpantiBmunchkinBmigranBmeinBmarshmallowBlurkBlipstickBkayakBjcBitttBhqBhattonBgroomBgrBfrisbeBfearlessBfatalBeughBendurBdoveBdomB	croissantBcoralinBchirpBcatherinBcanÃ¯BcaneBbuyerBbreedBbb10BauthentBarsenalBanoopBaltonB94B66B000BÃ«Bx3BwhaBwarcraftBwalaBvoyagBvoucherBvarietiBtwelvBtherelBstripperBscentBroamBrechargBreadinBpstBproxiBpplsBportionBplaguBnoobBnahhBmunchiBmergBmarinaBlollipopBlimoBhaaaBgreaterBglastonburiBghBfrickenBfetchBdustiBdoodlBdiplomaBdeedBdaydreamBdakotaBcomaBclumsiBcliffBcjBchocoBchicBcabinetBbubBboobiBbitchiBbenniBbartendBbarbecuBbahahahaBaxeBarcadBaltB12amBwbuBwahooBviolentBupliftB	unhealthiBtwiterBstudyinBstructurBstarterBsooooooooooBslBsamsungBruBrodBroBrefBprivaciBprickBpoolsidBpatrolBowiB
nottinghamB
noooooooooBnawwBmuchhBmoronBmofoBmitchBmilBmerchB	marsiscomBlovBlolllBlataBlagunaBhelicoptB	heartbeatBharleyBhairiBhahahaaBgivinBgelBeachBdevoBdawsonBcudntBcremeBcomedianBchowBcarrierBbecBbasiBbarfBbarbaraBbaddBasylmBassurBandrB140confB06BwhitneyBwhitBweirdestBvÃ£BvietnamBu2BtxtingBtobiBstinkinBsplendidBsnailBshaheenBserialBsconeBrpBriBrealistBpickupBphBobrienBmtgBmitBmgmtBmgBmanicBlynnBlumpBlouisBkirstiBkendraBinfinitBhormonBhighlandBhackerBgutterBgrimBgriffinBg2gBfuzzbalBfletcherBdsiBdoodBdistantBdireBdehydrBdebBdahBcueBcorniBclutchBclawBbriBbrekkiBbravoBbmwBblahhhBbjBalexandBÃ¹BÃ¸BwoohoooBwakeupBvmBvisiblB	unproductBunbearBtymBtrouserBtodayyyBtechiBt4BstylistBstubbornB	stockholmBshytBsemBseeeB	sandwhichBresolutBquadBpreachBpopsiclBoutaBordinariBokeyBoccupiBnooooooooooBnikonBmultiBmmwantBmananaBleightonBlandlordBkayaBislB	hoppusdayBhomoBhillariBharborBgymnastBfooBflyladiB	enlightenBdisordBdeltaBcoordinBcollaborBcaptionBbloatBbehalfBantiBanoBalterBakBacquirB93B86B330B02ByeaaahBwayyyiBwakinBvonBviolatBviewerBverifiBvaguBuhmmBtardBstakeBshanghaiB
shakespearB
sacramentoBromanBquidBquakBparcelBpakistanBmuteBmustardBmillerBmayoBlinkedinBlindseyBkelBjewelBjelloBisolB	inbetweenBhammockBgpaBglBfunnierBdualBdjingBdevinBdeliBdefinetBdawBcripplBcorkBcollinBcolaBchanelBcertB	bandwidthBapplebeBantiquBallllllBaliciaB3333B1130B05ByeahhhhBwbBvalBunicornBtideBthierBtakerBsweeetBsuperbBsummariBsthBsixthB	shareholdBsalariBrolandBreloadBrapperBpsychoBprawnBpoofBpersistBpendBos3BmyyBmunichBmottoBmishaBmeatbalBlizardB	literaturBlexBkanBiyaBinsectBhiveBhillsongBgrooviBgoofiBgeometriBfeeelBfaxBemailunlimitBellaBdratBdividendB	discouragBcs4BcrepeB	cranberriB	consciousBcomposBcollagBchroniclBchiefBcherishBcheetoBcandidBbumperBbrothaB	bronchitiBbraidBbiatchBbecuasBarvoBarrowBaddonB63B3hrsByooBveraB
twitterrifBtrayBtraumatBsydBsteadiBsophiaBsociologBshaqBscholarshipBsalutBredundBredesignBpervBonwardBnutellaBneeedBnayBministriBministBmeterBlancBl4dBjlsBjaiB	inventoriBhersheyB	hairsprayBguruBgucciBgoddessBfoilBfilthiBfifteenBexportB	dreamlandBconfigurBconcretBclanBcinciBcherylBchanBcapacBcanyonBbreakiBbordBbonjourBbeckBawwwwwwwBaaawB1kB1hrByoghurtBwooohoooBwheeBwaaaaayBuntouchBtonguetiBthroughBtf2BtensionBteheBsquadBroachBresignBreactBpolaroidBpittB	pinkberriBpadrBoutstandBnochBnhlBnappiBmowerBmovieBmayaBmalBkuyaBkickassBinnovBinducBhamptonBginaBgettnBflawBeu09BethicBeligBduvetBdoritoBdampBcyberBcrochetBcrackinBconsequBconradBchristiBbundlBbrittBbrettBbrasilB	blockbustBbittenBbeepBamazinggB711B645B3ohB2xB2gB228BÃ©ByeppByearoldBworldwidBwelpBwadeBupppBtntBtiredBteeniBtangoBspiltBsnifBshowtimBserenBsegmentBsedBresemblBredwBraisinBputerBpuneBpressiB	prescriptBpopulBparkerBpapiBopposBneicBmouseBmisplacBmilestonBmerriBluvinBkindergartenBkgB
javascriptBhawaiianBgeminiBfoolishBfebruariBfckB
enthusiasmBelitBeasiestBdugBdslBdriveinBdishwashBdawgBdaleBcucumbBcreedBcornwalBchoBchitownBcaterBcanoBcanberraBbusterBbrBbootlegBbetweenBbehaviorBbabygirlBauchBaboardBaawBzenByeshBxxxxxxxBworkkBworkiBwhoohooBwardBviennaBtotaliBtoasterBswBsquishB	satisfactBrollinBroarBrinBrileyBpretzelBphotogBphobiaBpalaBoutlinBnandoB	mccartneyBmartBmarkerBispBimhoBibizaBhttpskepticBhelloooBfresnoBfnBflorencBevolutBeurghBestimBentouragBduoBdeyBdeariBcruzBcldB	chickfilaBcatholBbetrayBbentBbachelorBaskinBarielBanythinB	amazingggBaggressBabsencB91B545BÃ BywBynBxddBworkdayBwolvBwahhBtumBtryoutBtiaBthÃ¯BthrBteleBstrainBstarshipBsrBsqlBsongzBsmirkontBslurpeBskullBshrinkB	shamelessBsandraBrihannaBrenoBrazorBrafaelB
queenslandBprosperBpppBporridgBosloB	orchestraBoraclBoffsprBnyaBnipBnedBnavigBmuddiB	minnesotaBmilliBmicroBmeyerBmarthaBmaccaBm8BlooolBllsBlesliBkristiBkimmelBhuzzahBhookerBharvestBhairstylBforrealBflockBferrelBentrancBdummiBdqBdonaldBdividBditoB
discontinuBdebugBdealerBcuzinB	companionBcolumbiaBchubbiBcarrBbonniBbonnarooBblossomBblimeyBaspectBairconBabortB77B59B10minB1015BzeldaByippiBwohooBwarehousBwaiterBvalentinBvaginaBunpleasBuninstalBtrunkBtroopBtextbookBtallerBswamBstlB
specialistBsoooonBsippinBshrugBshortiBshelvB
sharepointBseizurBscissorBruthBroyBrelivBrecognisBralphBqtBpshBprocedurBpriestBpresumBpoetBplasterBpeerBoutragBotwB	norwegianBnakBmuyBloveyBlottaBloomBkiddinB	judgementBjazziBjabBintolerB	interventBillusBhonestiBhiatusBgratBgentBfloBflexBfanficBerghBcougarBcooBcheckoutBchaserBchamberBboyzBboooooooBblazerBbelgianBavaBatticBarthurB
afterpartiB745B71B4hrsB4getBwoofBwebcastBvolBvendorBvendBunreadBugghhBthongBswanBsurBsuncreamBsuchaBstuntBstrayB
stomachachBspriteB	seventeenBscannerBrepoBprettierBplaceboBphiladelphiaBpeterfacinelliBnormBnipplBmidwestBmeuBmemoBmaplBmannerBmailboxBmadonnaBlpBlkBlicencB	legendariBjugglBjudeBjtvBircBinstinctBilovBhustlBhotmailBhorrifBhiphopBglitterBfundayBfloydBfanboyBextractBeeBeconBdustinBdilemmaBcrumblBcontagiBcomoBcolderBcircuitBchorusBcarlislBbroccoliBblechBbeverBbabyyBarkansaBzebraB	worthwhilB	worthlessBwoowBwhootBwaaaBunchartBtxtsBtwibeBtlkBtimerBtashaBsungBsummeriBstartrekBsqueeBsoderlBsnagB	sleeeeeepBsensiblBscandalBsaunaBsangriaBrochestBrewindBprodigiBprejudicBposerBplannerBpaoloBorgasmBomahaBnottBmphB
motherfuckBmorrowBmateyB	mandatoriBlottoBloftBlibertiBlcdBlargerBkentBkaratBituB
instructorBinaBillinoiBie6BhuBhorrifiBhistorBgreasiBgorgBfrostiBfirewalBfionaBexposurBdunoBdisposBdevotBdancinBcompassBcombatB	chihuahuaBchennaiBcanvaBbroadBbreakerBborinBbootcampBbdB
bankruptciBaviBatchaBafB74B61B350B31stB160B03BÃ­BzzzzzByourselfByhBxtraBwtBweeeeBwalliBwahhhBvinegarBvergB	venezuelaBtresBtminusBtenderBtediousBtapiBsunflowBspoiltBsoleBsnotBsmarterBslackerBskitBsidewalkBrhythmBremebBrehersBreceiptBphysioBpharmaciBpersuadB
parliamentB
pakcricketBowwwwBovercomBolympBnewerBnewayBnanoBmuggiBmarcoBmaciBmaaanBlidBkebabBkaraBjunoBinterstBinnoutB	inconveniBhÃ¯BholderBhikBgridBgreendayBgratitudBgenericBfurriBfirBevaluB	douchebagBdankBcussBchuBbeddBaucklandBapproxBaahhB200thB01BÃªByayyyyyBwebpagBvictorBversusB	unsuccessBtwBtortillaBtoastiBterracBtakeawayBspongB	spidermanBsoooooooooooBsoilBslipperBskydivBsensatBschBsaynowBsadiBrevelBqueriBpulBphotobucketBpastiB	paragraphBpacmanBorthodontistBomgshBmistiBmetricBmessinBmatrixBlvlBkolBkitaBkeywordBkennediBjudiBjgBjamaicaB
inappropriBimmBidcBiaBhootBhitch
??
Const_5Const*
_output_shapes	
:?>*
dtype0	*??
value??B??	?>"??                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_249586
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_249591
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	
signatures*
;

_lookup_layer
	keras_api
_adapt_function*
* 
* 
* 
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
7
lookup_table
token_counts
	keras_api*
* 
* 
* 

0*
* 
* 
* 
* 
R
_initializer
_create_resource
_initialize
_destroy_resource* 
?
_create_resource
_initialize
_destroy_resourceJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table*
* 
* 
* 
* 
* 
* 
* 
* 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_1
hash_tableConstConst_1Const_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_249470
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1Const_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_249628
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameMutableHashTable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_249641??
?a
?
!__inference__wrapped_model_249063
input_1^
Zsequential_1_text_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handle_
[sequential_1_text_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value	;
7sequential_1_text_vectorization_string_lookup_1_equal_y>
:sequential_1_text_vectorization_string_lookup_1_selectv2_t	
identity	??Msequential_1/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2l
+sequential_1/text_vectorization/StringLowerStringLowerinput_1*'
_output_shapes
:??????????
2sequential_1/text_vectorization/StaticRegexReplaceStaticRegexReplace4sequential_1/text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
'sequential_1/text_vectorization/SqueezeSqueeze;sequential_1/text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????r
1sequential_1/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
9sequential_1/text_vectorization/StringSplit/StringSplitV2StringSplitV20sequential_1/text_vectorization/Squeeze:output:0:sequential_1/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
?sequential_1/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Asequential_1/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Asequential_1/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
9sequential_1/text_vectorization/StringSplit/strided_sliceStridedSliceCsequential_1/text_vectorization/StringSplit/StringSplitV2:indices:0Hsequential_1/text_vectorization/StringSplit/strided_slice/stack:output:0Jsequential_1/text_vectorization/StringSplit/strided_slice/stack_1:output:0Jsequential_1/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Asequential_1/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csequential_1/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csequential_1/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;sequential_1/text_vectorization/StringSplit/strided_slice_1StridedSliceAsequential_1/text_vectorization/StringSplit/StringSplitV2:shape:0Jsequential_1/text_vectorization/StringSplit/strided_slice_1/stack:output:0Lsequential_1/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Lsequential_1/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
bsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastBsequential_1/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastDsequential_1/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapefsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
ksequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdusequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0usequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
psequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatertsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ysequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
ksequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastrsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxfsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0wsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
lsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ssequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0usequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulosequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumhsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumhsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
nsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
osequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountfsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0wsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
isequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumvsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
msequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
isequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2vsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0jsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0rsequential_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Msequential_1/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Zsequential_1_text_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handleBsequential_1/text_vectorization/StringSplit/StringSplitV2:values:0[sequential_1_text_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
5sequential_1/text_vectorization/string_lookup_1/EqualEqualBsequential_1/text_vectorization/StringSplit/StringSplitV2:values:07sequential_1_text_vectorization_string_lookup_1_equal_y*
T0*#
_output_shapes
:??????????
8sequential_1/text_vectorization/string_lookup_1/SelectV2SelectV29sequential_1/text_vectorization/string_lookup_1/Equal:z:0:sequential_1_text_vectorization_string_lookup_1_selectv2_tVsequential_1/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
8sequential_1/text_vectorization/string_lookup_1/IdentityIdentityAsequential_1/text_vectorization/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:?????????~
<sequential_1/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
4sequential_1/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
Csequential_1/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor=sequential_1/text_vectorization/RaggedToTensor/Const:output:0Asequential_1/text_vectorization/string_lookup_1/Identity:output:0Esequential_1/text_vectorization/RaggedToTensor/default_value:output:0Dsequential_1/text_vectorization/StringSplit/strided_slice_1:output:0Bsequential_1/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentityLsequential_1/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:??????????
NoOpNoOpN^sequential_1/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
Msequential_1/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2Msequential_1/text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_sequential_1_layer_call_fn_249130
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_249119o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?W
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_249273
input_1Q
Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_1_equal_y1
-text_vectorization_string_lookup_1_selectv2_t	
identity	??@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2_
text_vectorization/StringLowerStringLowerinput_1*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
(text_vectorization/string_lookup_1/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_1_equal_y*
T0*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/SelectV2SelectV2,text_vectorization/string_lookup_1/Equal:z:0-text_vectorization_string_lookup_1_selectv2_tItext_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/IdentityIdentity4text_vectorization/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:04text_vectorization/string_lookup_1/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:??????????
NoOpNoOpA^text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?W
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_249325
input_1Q
Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_1_equal_y1
-text_vectorization_string_lookup_1_selectv2_t	
identity	??@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2_
text_vectorization/StringLowerStringLowerinput_1*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
(text_vectorization/string_lookup_1/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_1_equal_y*
T0*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/SelectV2SelectV2,text_vectorization/string_lookup_1/Equal:z:0-text_vectorization_string_lookup_1_selectv2_tItext_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/IdentityIdentity4text_vectorization/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:04text_vectorization/string_lookup_1/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:??????????
NoOpNoOpA^text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?W
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_249197

inputsQ
Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_1_equal_y1
-text_vectorization_string_lookup_1_selectv2_t	
identity	??@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
(text_vectorization/string_lookup_1/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_1_equal_y*
T0*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/SelectV2SelectV2,text_vectorization/string_lookup_1/Equal:z:0-text_vectorization_string_lookup_1_selectv2_tItext_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/IdentityIdentity4text_vectorization/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:04text_vectorization/string_lookup_1/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:??????????
NoOpNoOpA^text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
/
__inference__initializer_249546
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference_save_fn_249570
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?W
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_249403

inputsQ
Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_1_equal_y1
-text_vectorization_string_lookup_1_selectv2_t	
identity	??@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
(text_vectorization/string_lookup_1/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_1_equal_y*
T0*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/SelectV2SelectV2,text_vectorization/string_lookup_1/Equal:z:0-text_vectorization_string_lookup_1_selectv2_tItext_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/IdentityIdentity4text_vectorization/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:04text_vectorization/string_lookup_1/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:??????????
NoOpNoOpA^text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
-__inference_sequential_1_layer_call_fn_249221
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_249197o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
G
__inference__creator_249541
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_55227*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
;
__inference__creator_249523
identity??
hash_tablem

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name57752*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
-
__inference__destroyer_249551
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__traced_save_249628
file_prefixJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?BFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1savev2_const_6"/device:CPU:0*
_output_shapes
 *
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
::

_output_shapes
::

_output_shapes
: 
?
+
__inference_<lambda>_249591
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
-__inference_sequential_1_layer_call_fn_249351

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_249197o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
-
__inference__destroyer_249536
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference_<lambda>_2495868
4key_value_init57751_lookuptableimportv2_table_handle0
,key_value_init57751_lookuptableimportv2_keys2
.key_value_init57751_lookuptableimportv2_values	
identity??'key_value_init57751/LookupTableImportV2?
'key_value_init57751/LookupTableImportV2LookupTableImportV24key_value_init57751_lookuptableimportv2_table_handle,key_value_init57751_lookuptableimportv2_keys.key_value_init57751_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init57751/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?>:?>2R
'key_value_init57751/LookupTableImportV2'key_value_init57751/LookupTableImportV2:!

_output_shapes	
:?>:!

_output_shapes	
:?>
?
?
__inference_restore_fn_249578
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
?
-__inference_sequential_1_layer_call_fn_249338

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_249119o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?C
?
__inference_adapt_step_249518
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?
?
"__inference__traced_restore_249641
file_prefixM
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: 

identity_1??2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?BFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2	?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:0RestoreV2:tensors:1*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
IdentityIdentityfile_prefix3^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: S

Identity_1IdentityIdentity:output:0^NoOp_1*
T0*
_output_shapes
: }
NoOp_1NoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
?W
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_249455

inputsQ
Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_1_equal_y1
-text_vectorization_string_lookup_1_selectv2_t	
identity	??@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
(text_vectorization/string_lookup_1/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_1_equal_y*
T0*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/SelectV2SelectV2,text_vectorization/string_lookup_1/Equal:z:0-text_vectorization_string_lookup_1_selectv2_tItext_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/IdentityIdentity4text_vectorization/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:04text_vectorization/string_lookup_1/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:??????????
NoOpNoOpA^text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?W
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_249119

inputsQ
Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*text_vectorization_string_lookup_1_equal_y1
-text_vectorization_string_lookup_1_selectv2_t	
identity	??@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mtext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ntext_vectorization_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
(text_vectorization/string_lookup_1/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0*text_vectorization_string_lookup_1_equal_y*
T0*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/SelectV2SelectV2,text_vectorization/string_lookup_1/Equal:z:0-text_vectorization_string_lookup_1_selectv2_tItext_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
+text_vectorization/string_lookup_1/IdentityIdentity4text_vectorization/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:04text_vectorization/string_lookup_1/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:??????????
NoOpNoOpA^text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2@text_vectorization/string_lookup_1/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_signature_wrapper_249470
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_249063o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference__initializer_2495318
4key_value_init57751_lookuptableimportv2_table_handle0
,key_value_init57751_lookuptableimportv2_keys2
.key_value_init57751_lookuptableimportv2_values	
identity??'key_value_init57751/LookupTableImportV2?
'key_value_init57751/LookupTableImportV2LookupTableImportV24key_value_init57751_lookuptableimportv2_table_handle,key_value_init57751_lookuptableimportv2_keys.key_value_init57751_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init57751/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?>:?>2R
'key_value_init57751/LookupTableImportV2'key_value_init57751/LookupTableImportV2:!

_output_shapes	
:?>:!

_output_shapes	
:?>"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????H
text_vectorization2
StatefulPartitionedCall_1:0	?????????tensorflow/serving/predict:?6
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	
signatures"
_tf_keras_sequential
P

_lookup_layer
	keras_api
_adapt_function"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_sequential_1_layer_call_fn_249130
-__inference_sequential_1_layer_call_fn_249338
-__inference_sequential_1_layer_call_fn_249351
-__inference_sequential_1_layer_call_fn_249221?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_1_layer_call_and_return_conditional_losses_249403
H__inference_sequential_1_layer_call_and_return_conditional_losses_249455
H__inference_sequential_1_layer_call_and_return_conditional_losses_249273
H__inference_sequential_1_layer_call_and_return_conditional_losses_249325?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_249063input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
serving_default"
signature_map
L
lookup_table
token_counts
	keras_api"
_tf_keras_layer
"
_generic_user_object
?2?
__inference_adapt_step_249518?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_249470input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
j
_initializer
_create_resource
_initialize
_destroy_resourceR jCustom.StaticHashTable
O
_create_resource
_initialize
_destroy_resourceR Z
table
"
_generic_user_object
"
_generic_user_object
?2?
__inference__creator_249523?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_249531?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_249536?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_249541?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_249546?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_249551?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_249570checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_249578restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_57
__inference__creator_249523?

? 
? "? 7
__inference__creator_249541?

? 
? "? 9
__inference__destroyer_249536?

? 
? "? 9
__inference__destroyer_249551?

? 
? "? @
__inference__initializer_249531#$?

? 
? "? ;
__inference__initializer_249546?

? 
? "? ?
!__inference__wrapped_model_249063? !0?-
&?#
!?
input_1?????????
? "G?D
B
text_vectorization,?)
text_vectorization?????????	j
__inference_adapt_step_249518I"??<
5?2
0?-?
??????????IteratorSpec 
? "
 z
__inference_restore_fn_249578YK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_249570?&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
H__inference_sequential_1_layer_call_and_return_conditional_losses_249273g !8?5
.?+
!?
input_1?????????
p 

 
? "%?"
?
0?????????	
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_249325g !8?5
.?+
!?
input_1?????????
p

 
? "%?"
?
0?????????	
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_249403f !7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????	
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_249455f !7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????	
? ?
-__inference_sequential_1_layer_call_fn_249130Z !8?5
.?+
!?
input_1?????????
p 

 
? "??????????	?
-__inference_sequential_1_layer_call_fn_249221Z !8?5
.?+
!?
input_1?????????
p

 
? "??????????	?
-__inference_sequential_1_layer_call_fn_249338Y !7?4
-?*
 ?
inputs?????????
p 

 
? "??????????	?
-__inference_sequential_1_layer_call_fn_249351Y !7?4
-?*
 ?
inputs?????????
p

 
? "??????????	?
$__inference_signature_wrapper_249470? !;?8
? 
1?.
,
input_1!?
input_1?????????"G?D
B
text_vectorization,?)
text_vectorization?????????	