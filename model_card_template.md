# Model Card: Income Classification Model

## Model Details
This **Gradient Boosting Classifier** model is designed to predict whether an individual's income exceeds $50,000/year based on 1994 U.S. Census data. The model was trained using Scikit-Learn's GradientBoostingClassifier with the following specifications:

- **Algorithm**: `GradientBoostingClassifier`
- **Hyperparameters**: `n_estimators=100`, `random_state=42`
- **Training Date**: October 2023
- **Framework**: Scikit-Learn 1.5.1/1.3.2

## Intended Use
**Educational and demonstration purposes only.** This model was created for a machine learning engineering course project. 

**Do NOT use for:**
- Real financial decisions
- Employment screening
- Credit scoring
- Any production systems without extensive ethical review

## Training Data
### Dataset
1994 U.S. Census Bureau dataset (`data/census.csv`):
- **Total Rows**: 32,561
- **Features**: 14
- **Target**: `salary` (binary: <=50K or >50K)

### Features
**Categorical (8 features):**
- `workclass`
- `education`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native-country`

**Numerical (6 features):**
- `age`
- `fnlgt` (fnlwgt)
- `education-num`
- `capital-gain`
- `capital-loss`
- `hours-per-week`

### Data Processing
1. **Missing Values**: 4,262 rows with "?" handled by OneHotEncoder with `handle_unknown="ignore"`
2. **Encoding**: Categorical features encoded via OneHotEncoder
3. **Label Processing**: LabelBinarizer converts salary to binary (0/1)
4. **Train/Test Split**: 80/20 split with `random_state=42`

## Evaluation Data
- **Test Set**: 20% of original data (~6,512 instances)
- **Split Method**: `train_test_split` with `test_size=0.2`, `random_state=42`

## Metrics
### Overall Performance
- **Precision**: 0.8030
- **Recall**: 0.6225
- **F1 Score**: 0.7013

### Performance on Data Slices
**Key Findings from `slice_output.txt`:**

**Workclass:**
- Private (4,578 samples): **Precision 0.8222** | **Recall 0.6004** | **F1 0.6940**
- Self-emp-inc (212 samples): **Precision 0.7982** | **Recall 0.7712** | **F1 0.7845**
- Local-gov (387 samples): **Precision 0.7925** | **Recall 0.7636** | **F1 0.7778**

**Education:**
- Doctorate (77 samples): **Precision 0.8361** | **Recall 0.8947** | **F1 0.8644**
- Masters (369 samples): **Precision 0.8381** | **Recall 0.8502** | **F1 0.8441**
- HS-grad (2,085 samples): **Precision 0.9118** | **Recall 0.2696** | **F1 0.4161**

**Gender:**
- Male (4,387 samples): **Precision 0.8037** | **Recall 0.6428** | **F1 0.7143**
- Female (2,126 samples): **Precision 0.7973** | **Recall 0.5064** | **F1 0.6194**

**Race:**
- White (5,595 samples): **Precision 0.8056** | **Recall 0.6211** | **F1 0.7015**
- Black (599 samples): **Precision 0.8000** | **Recall 0.5538** | **F1 0.6545**
- Asian-Pac-Islander (193 samples): **Precision 0.7667** | **Recall 0.7419** | **F1 0.7541**

## Ethical Considerations
### Documented Disparities
1. **Gender Bias**: Female recall (0.5064) is 21.3% lower than male recall (0.6428)
2. **Racial Disparities**: F1 scores vary by up to 0.10 between racial groups
3. **Education Bias**: Model performs best on highly educated individuals

### Recommendations
1. **Bias Mitigation**: Address gender and racial disparities before production use
2. **Continuous Monitoring**: Track performance across protected groups
3. **Transparency**: Document all limitations clearly

## Caveats and Recommendations
### Limitations
1. **Historical Data**: Trained on 1994 data - economic conditions have changed
2. **U.S.-Centric**: Primarily represents 1994 U.S. demographics
3. **Binary Classification**: Simplifies complex income distribution
4. **Missing Data**: 4,262 rows contain "?" values

### Technical Recommendations
1. **Regular Updates**: Retrain with contemporary data
2. **Bias Testing**: Implement fairness audits
3. **Feature Scaling**: Add scaling for continuous features
4. **Cross-Validation**: Use k-fold CV for more reliable metrics

### Usage Warnings
1. **Not Production-Ready**: Educational model only
2. **Bias Concerns**: Documented disparities require mitigation
3. **Data Recency**: 30+ year old data

## Model Artifacts
```bash
model/model.pkl           # Trained GradientBoostingClassifier
model/encoder.pkl         # Fitted OneHotEncoder
model/lb.pkl             # Fitted LabelBinarizer
slice_output.txt         # Complete slice performance metrics