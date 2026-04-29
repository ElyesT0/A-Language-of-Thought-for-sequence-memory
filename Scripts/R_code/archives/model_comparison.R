library(lme4)
data = read.csv("/Users/elyestabbane/Documents/UNICOG/2-Experiments/memocrush/Data/processed/both/two_experiment_datasets_2024-06_18.csv")
data3 = subset(data,seq_name=="3items_Repetition" | seq_name=="control_3items_Repetition")
data3.model_regularite_temporelle = lmer(distance_dl ~ seq_name + (1|participant_ID),data = data3)
data3.model = lmer(distance_dl ~ seq_name + no_rotation + (1|participant_ID),data = data3)
anova(data3.model_regularite_temporelle,data3.model)

data3 = subset(data,seq_name=="3items_Repetition" | seq_name=="control_3items_Repetition"| seq_name=="Repetition_nested"| seq_name=="control_NoLocal_nested" |seq_name=="control_NoGlobal_nested"  )
data3.model = lmer(distance_dl ~ LoT_complexity + no_rotation + (1|participant_ID)+LoT_complexity*no_rotation,data = data3)
data3.model_gt = lmer(distance_dl ~ LoT_complexity + (1|participant_ID),data = data3)
anova(data3.model_gt,data3.model)


boxplot(distance_dl ~ LoT Complexity,
        col=c("white","lightgray"),data)
boxplot(distance_dl ~ complexity_chunk_group,
        col=c("white","lightgray"),data)
boxplot(distance_dl ~ complexity_chunk_item,
        col=c("white","lightgray"),data)
boxplot(distance_dl ~ complexity_lempel_ziv,
        col=c("white","lightgray"),data)
boxplot(distance_dl ~ complexity_shannon_entropy,
        col=c("white","lightgray"),data)

# nouvelle mesure complexité : dl("AAAABBBBCCCC","ABCABCABCABC")
# We know want to observe how well distance_dl (our best measure for performance) can be explained by complexity_chunk_group
data.model_LoT=lmer(distance_dl~LoT_complexity + (1|participant_ID),data=data)
data.model_LoT=lmer(distance_dl~LoT_complexity + (1|participant_ID),data=data)
AIC_value_LoT <- AIC(data.model_LoT)

# We know want to observe how well distance_dl (our best measure for performance) can be explained by complexity_chunk_group
data.model_chunkGroup=lmer(distance_dl~complexity_chunk_group + (1|participant_ID),data=data)
AIC_value_chunkGroup <- AIC(data.model_chunkGroup)
# We know want to observe how well distance_dl (our best measure for performance) can be explained by complexity_chunk_item
data.model_Item=lmer(distance_dl~complexity_chunk_item + (1|participant_ID),data=data)
AIC_value_chunkItem <- AIC(data.model_chunkItem)
# We know want to observe how well distance_dl (our best measure for performance) can be explained by complexity_lempel_ziv
data.model_LempelZiv=lmer(distance_dl~complexity_lempel_ziv + (1|participant_ID),data=data)
AIC_value_LempelZiv<- AIC(data.model_LempelZiv)
# We know want to observe how well distance_dl (our best measure for performance) can be explained by complexity_shannon_entropy
data.model_ShannonEntropy=lmer(distance_dl~complexity_shannon_entropy + (1|participant_ID),data=data)
AIC_value_ShannonEntropy<- AIC(data.model_ShannonEntropy)
# We know want to observe how well distance_dl (our best measure for performance) can be explained by DL_distance to simplest 
data.model_dlToSimplest=lmer(distance_dl~dl_toSimplest_comp + (1|participant_ID),data=data)
AIC_value_dlToSimplest<- AIC(data.model_dlToSimplest)
