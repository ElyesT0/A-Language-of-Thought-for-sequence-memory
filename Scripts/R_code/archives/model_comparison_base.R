library(lme4)
data = read.csv("/Users/elyestabbane/Documents/UNICOG/2-Experiments/memocrush/Data/processed/both/2024-08_22_complexity_dataset_EXT_only_REP.csv")


boxplot(distance_dl ~ LoT.Complexity,
        col=c("white","lightgray"),data)
boxplot(distance_dl ~ Subjective_complexity,
        col=c("white","lightgray"),data)
boxplot(distance_dl ~ Shannon.Entropy,
        col=c("white","lightgray"),data)
boxplot(distance_dl ~ Lempel_Ziv,
        col=c("white","lightgray"),data)
boxplot(distance_dl ~ Change.Complexity,
        col=c("white","lightgray"),data)
boxplot(distance_dl ~ Algorithmic.Complexity,
        col=c("white","lightgray"),data)
boxplot(distance_dl ~ Subsymetries,
        col=c("white","lightgray"),data)
boxplot(distance_dl ~ Chunk.Complexity,
        col=c("white","lightgray"),data)


data.model_LoT=lmer(distance_dl~LoT.Complexity + (1|participant_ID),data=data)
AIC_value_LoT <- AIC(data.model_LoT)
print(sprintf('AIC_value_LoT: %f', AIC_value_LoT))

data.Subjective_complexity=lmer(distance_dl~Subjective_complexity + (1|participant_ID),data=data)
AIC_value_subjective <- AIC(data.Subjective_complexity)
print(sprintf('AIC_value_subjective: %f', AIC_value_subjective))


# We now want to observe how well distance_dl (our best measure for performance) can be explained by complexity_chunk
data.ShannonEntropy=lmer(distance_dl~Shannon.Entropy + (1|participant_ID),data=data)
AIC_value_ShannonEntropy <- AIC(data.ShannonEntropy)
print(sprintf('AIC_value_ShannonEntropy: %f', AIC_value_ShannonEntropy))

# We now want to observe how well distance_dl (our best measure for performance) can be explained by complexity_lempel_ziv
data.model_LempelZiv=lmer(distance_dl~Lempel_Ziv + (1|participant_ID),data=data)
AIC_value_LempelZiv<- AIC(data.model_LempelZiv)
print(sprintf('AIC_value_LempelZiv: %f', AIC_value_LempelZiv))

# We now want to observe how well distance_dl (our best measure for performance) can be explained by change complexity
data.model_Change=lmer(distance_dl~Change.Complexity + (1|participant_ID),data=data)
AIC_value_change<- AIC(data.model_Change)
print(sprintf('AIC_value_change: %f', AIC_value_change))

# We now want to observe how well distance_dl (our best measure for performance) can be explained by Algorithmic.Complexity
data.algorithmic=lmer(distance_dl~Algorithmic.Complexity + (1|participant_ID),data=data)
AIC_value_algorithmic <- AIC(data.algorithmic)
print(sprintf('AIC_value_algorithmic: %f', AIC_value_algorithmic))

# We now want to observe how well distance_dl (our best measure for performance) can be explained by subsymetries
data.model_Subsymetries=lmer(distance_dl~Subsymetries + (1|participant_ID),data=data)
AIC_value_Subsymetries<- AIC(data.model_Subsymetries)
print(sprintf('AIC_value_Subsymetries: %f', AIC_value_Subsymetries))

# We now want to observe how well distance_dl (our best measure for performance) can be explained by complexity_chunk
data.chunk=lmer(distance_dl~Chunk.Complexity + (1|participant_ID),data=data)
AIC_value_chunk <- AIC(data.chunk)
print(sprintf('AIC_value_chunk: %f', AIC_value_chunk))
