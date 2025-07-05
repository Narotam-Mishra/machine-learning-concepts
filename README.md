
## AI Vs ML vs DL vs Data Science (00:01:25 )

### **1. Artificial Intelligence (AI)**  
- AI involves creating applications that perform tasks **without human intervention**.  
- Examples:  
  - **Netflix recommendations** (suggests movies based on viewing history).  
  - **Amazon product recommendations** (suggests related items like headphones after buying a phone).  
  - **Self-driving cars** (Tesla uses AI to navigate roads autonomously).  

### **2. Machine Learning (ML) – Subset of AI**  
- Provides **statistical tools** to:  
  - Analyze & visualize data.  
  - Make **predictions/forecasts** (e.g., sales forecasting, fraud detection).  
- Relies on **algorithms with statistical techniques** (e.g., regression, decision trees).  

### **3. Deep Learning (DL) – Subset of ML**  
- Mimics the **human brain** using **multi-layered neural networks**.  
- Solves **complex problems** (e.g., image recognition, speech processing).  
- Enabled breakthroughs in AI (e.g., ChatGPT, advanced computer vision).  

### **4. Data Science – The Bigger Picture**  
- A **data scientist** works across **AI, ML, and DL** depending on the business need.  
- Tasks may include:  
  - **Data analysis** (PowerBI, SQL).  
  - **ML/DL modeling** (predictive algorithms, neural networks).  
- Goal: **Build AI-driven solutions** for real-world problems.  

### **Key Takeaways**  
- **AI** → Autonomous decision-making.  
- **ML** → Stats-based predictions (subset of AI).  
- **DL** → Brain-inspired neural networks (subset of ML).  
- **Data Science** → Umbrella field combining all three for business solutions.  

- AI is a process wherein we create some kind of applications in which it will be able to do its task without any human intervention where it'll be able to make decisions and perfrom its tasks. Example - Netflix's Recommendation System, Self Driven Car (Tesla),

- Machine Learning is subset of AI where it provides stats tools to analyze the data, visualize the data and apart from that it used do predictions and forcasting.

- Deep Learning is a subset of Machine Learning. It's main aim is to mimic human brain so they actually create multi-layered neural network and this multi layered neural network will basically help us to train the machines or apps whatever we will try to create.

### Basic concepts of Neural Network
- A neural network is a computational model inspired by how biological neurons in the brain process information. It consists of interconnected nodes (neurons) organized in layers that can learn patterns from data.

**Basic Structure:**
- **Input Layer**: Receives data (like pixel values of an image)
- **Hidden Layer(s)**: Process and transform the data
- **Output Layer**: Produces the final result

**How it works:**
Each connection between neurons has a "weight" that determines how much influence one neuron has on another. During training, these weights are adjusted to minimize errors and improve predictions.

**Simple Examples:**

1. **Image Recognition**: A neural network can identify if an image contains a cat or dog by learning features like edges, shapes, and textures through its layers.

2. **Email Spam Detection**: The network learns to recognize patterns in email text (certain words, phrases, sender patterns) to classify emails as spam or legitimate.

3. **Handwriting Recognition**: When you write on your phone, neural networks recognize the shapes and strokes to convert your handwriting into text.

4. **Voice Assistants**: Neural networks process sound waves from your speech, convert them to text, understand the meaning, and generate appropriate responses.

The "deep" in deep learning refers to having many hidden layers (typically 3 or more), allowing the network to learn increasingly complex patterns and representations from the data.

## Machine Learning and Deep Learning (00:07:56)

1. **Supervised Machine Learning** Uses labeled training data (input-output pairs) to learn patterns and make predictions. The algorithm learns from examples where the correct answer is provided. Examples include email spam detection, image classification, and price prediction. (Uses labeled data for training)  
   - **Regression**: Predicts **continuous values** (e.g., predicting weight based on age).  
   - **Classification**: Predicts **discrete categories** (e.g., spam vs. not spam).  

2. **Unsupervised Machine Learning** Works with unlabeled data to discover hidden patterns or structures without knowing the "correct" answers. The algorithm finds relationships in data on its own. Examples include customer segmentation, anomaly detection, and data compression. (No labels, finds hidden patterns)  
   - **Clustering**: Groups similar data (e.g., customer segmentation).  
   - **Dimensionality Reduction**: Reduces features while keeping key info (e.g., Principle Component Analysis (PCA)).

### Key Difference: 
Supervised learning learns from examples with known outcomes, while unsupervised learning finds patterns in data without being told what to look for. 

In case of Unsupervised learning, there is no dependent variable (output)

### **Example (Supervised Learning)**  
- **Dataset**: Age (input) → Weight (output).  
- If predicting **weight (numeric)** → **Regression**.  
- If predicting **weight category (e.g., underweight/healthy/overweight)** → **Classification**.  

- With respect to any kind of problem statement that we solve, the majority of the business use cases will be fall under two sections :-
1. Supervised ML :- In case, we will mainly solve two major problem statements 
    - Regression problem
    - Classification 

2. Unsupervised ML :- In this case, we will solve two kind of problems :-
    - Clustering
    - Dimensionality Reduction

## Regression And Classification (00:09:05)

#### **1. Classification (Supervised ML)**  
- **Problem Type**: Predicts **categorical outcomes** (discrete labels).  
  - **Binary Classification**: 2 outcomes (e.g., Pass/Fail based on study/sleep/play hours).  
  - **Multiclass Classification**: >2 outcomes (e.g., Grades A/B/C).  
- **Example**: Predict if a student **passes/fails** using study/sleep/play data.  

#### **2. Clustering (Unsupervised ML)**  
- **Goal**: Group similar data points **without labels**.  
- **Example**: **Customer segmentation** using Salary vs. Age data.  
  - Groups: "High-income young professionals," "Middle-class," etc.  
  - **Application**: Targeted ads (e.g., luxury products for high-income clusters).  

#### **3. Dimensionality Reduction (Unsupervised ML)**  
- **Goal**: Reduce features (e.g., 1000 → 100) while retaining key info.  
- **Algorithms**: PCA (Principal Component Analysis), LDA.  
- **Use Case**: Simplifying complex datasets for faster processing.  

---

### ****Supervised ML** Algorithms Covered**    
1. Linear Regression  
2. Ridge & Lasso Regression  
3. Logistic Regression  
4. Decision Trees (Classification & Regression)  
5. AdaBoost  
6. Random Forest  
7. Gradient Boosting  
8. XGBoost  
9. Naive Bayes  
10. SVM (Support Vector Machines)  
11. KNN (K-Nearest Neighbors)  

#### **Unsupervised ML Algorithms**:  
1. K-Means Clustering  
2. DBSCAN  
3. Hierarchical Clustering  
4. K Nearest Neighbour Clustering
5. PCA (Dimensionality Reduction)  
6. LDA (Linear Discriminant Analysis)  

---

### **Key Takeaways**  
- **Classification** → Predict categories (Pass/Fail).  
- **Clustering** → Group unlabeled data (Customer segments).  
- **Dimensionality Reduction** → Compress features (PCA).  
- **Next**: Dive into **Linear Regression** (first algorithm).  

- Whenever we will solve a problem in case of supervised machine learning, there will be one dependent feature and there can be any number of independent features. For Example - In case of Age-Weight example, we can consider `age` (which is taken as input) as independent feature and `weight` (that is output) as dependent feature. 

- In case of Regression problem, the output will be continuous variable 

- In case of Classification problem, there will be fixed number of categories in the output of problem.

### Clustering and Dimensionality Reduction

- Clustering :- In case of clustering, we used to group similar data. Example - Ad-Marketing uses clustering through Customer Segmentation

## Linear Regression Algorithm (00:18:14)
![Linear Regression](/notes/01_Linear_Regression_1.1.png)

**Linear Regression Problem Statement** - Suppose we have given dataset of age (on x-axis) & weight (on y-axis) on 2D plane of XY, where we have calculated weight on the basis of age. In case of Linear Regression, we will try to find best fit line which will help us to do the predicition, i.e wrt any new age (on x-axis) what will be output (weight on y-axis). So, Y-axis (weight) is linear function of X-axis (age)

In case of Linear Regression we try to create a model with the help of training dataset, where the model (hypothesis) takes new age (independent feature) and gives the output of weight and with the help of performance metrics we try to verify whether that model is performing well or not.

![Equation of a straight line](/notes/01_Linear_Regression_Equation_of_Straight_line_1.2.png)