# Introduction to statistical learning and applications 2024

--- 

### About the final exam

The final exam will happen on **Friday 09-May at 8am**. It will be a 3h practical evaluation on the computers from ENSIMAG. It will have some multiple-choice questions to evaluate your theoretical knowledge and some practical questions similar to what you will do in the TPs. You will not have access to any internet connections, but you can bring your own handwritten and printed notes. In the computer that you will use, we will provide you a folder with the instructions for the exam, the `ipynb` files to fill with your answers and code, as well some materials of the course.

You are all expected to have read the scientific article : 

> Mark Newman  "Modularity and community structure in networks"
> https://arxiv.org/abs/physics/0602124

--- 

Hi :wave: 

This is the main page to look for materials of our classes on " Statistical learning and applications" in 2025. Please note that we **won't be using Chamilo**, 
so you should be checking this page regularly for updates.

### Instructors
- Pedro L. C. Rodrigues — pedro.rodrigues@inria.fr (CM, TD, and TP)
- Alexandre Wendling — alexandre.wendling@univ-grenoble-alpes.fr (TD and TP)
- Razan Mhanna — razan.mhanna@inria.fr (Complementary courses for M1AM students)

### Schedule
We will meet on Mondays from 15h30 to 18h30. Each time we will have a 1h30 CM 
and a 1h30 TD or TP session. 

We will be doing a total of 11 CM, 6 TP sessions, and 6 TD sessions.

The (initial) plan for the topics of each of our classes is the following:

- **Week 1 (03-February 2025)** 
  - CM1: Introduction, simple linear regression, multivariate statistics
  - TD1: Exercises on multivariate statistics and regression

- **Week 2 (10-February 2025)** 
  - CM2: Multiple linear regression
  - TP1: Regression

- **Week 3.1 (17-February 2025)**
  - CM3: Cross-validation, model selection, bias-variance tradeoff
  - TP1: Regression -- **Deadline 09-March at 23h59**

- **Week 3.2 (21-February 2025, 17-18h30)**
  - TD2: Some questions from previous exams

- **Week 4.1 (24-February 2025)**
  - CM4: Principal component analysis
  - TD3: Exercises on linear regression and PCA

- **Week 4.2 (only Pedro's group : 28-February 2025, 17-18h30)**
  - TD4: Some questions from previous exams + Help with TP1

- **Week 5 (10-March 2025)**
  - CM5: Linear classification — Discriminative approach
  - TP2: Principal components regression (PCR)
  
- **Week 6 (17-March 2025)**
  - CM6: Linear classification — Generative approach
  - TP2: Principal components regression (PCR) -- **Deadline 23-March at 23h59**

- **Week 7 (24-March 2025)**
  - CM7: Decision trees
  - TP3: Benchmarking classification methods
  
- **Week 8 (31-March 2025)**
  - CM8: Ensemble methods
  - TP3: Benchmarking classification methods -- **Deadline 14-Avril at 23h59**

- **Week 9.1 (07-April 2025)**
  - CM9: ML competitions, metrics, etc.
  - TD5: Gradient boosting

- **Week 9.2 (11-April 2025, 17-18h30)**
  - CM10: Introduction to network analysis

- **Week 10.1 (14-April 2025)**
  - CM11: Community detection in graphs
  - TD6: Some questions from previous exams

- **Week 10.2 (only Alexandre's group : 18-April 2025, 17-18h30)**
  - TD7: Some questions from previous exams

### Textbooks
- Our course has an official booklet available [here](https://cloud.univ-grenoble-alpes.fr/s/iTtXPTdLpyMwBtN).
    - **Important**: Please note that although this is the “official” booklet, the materials covered in class are not entirely included in it. Also, I won’t be presenting everything that is written in the booklet. Conclusion: to be sure of what I’ve given in class, you should always come to classes! (and take a look at the slides)
- Many examples and explanations given in class were inspired (or unashamedly copied) from Cosma Shalizi’s excellent lecture notes *“The truth about linear regression”* available [here](https://www.stat.cmu.edu/~cshalizi/TALR/).
- Students should also consider reading chapters from James et al. (2022) *"Introduction to statistical learning with applications to Python"* which is freely (and legally) available [here](https://www.statlearning.com/).
- Another excellent reference is Hastie et al. (2017) *"Elements of Statistical Learning"* which is freely (and legally) available [here](https://hastie.su.domains/ElemStatLearn/).

### Grading
Your final grade will the average of your TP scores (50%) and the final exam (50%)

#### -- Concerning the TP scores
You will be doing three TPs during this class and your final TP score will be based on their average score. Please be sure to do your reports properly and nicely, always including your `ipynb` file in your final submission so your professors can rerun your experiments and code. You should always include the name of all students from your group in your final report. The report should be delivered as a `zip` file on [Teide](https://teide.ensimag.fr/). If no one in your team has access to it, then you can send your report by e-mail to [Pedro](mailto:pedro.rodrigues@inria.fr) or [Alexandre](mailto:alexandre.wendling@univ-grenoble-alpes.fr), depending on which TP group you are in.

#### -- Concerning the final exam
The final exam will be a 3h practical evaluation on the computers from ENSIMAG. It will have some multiple-choice questions to evaluate your theoretical knowledge and some practical questions similar to what you will do in the TPs. You will not have access to any internet connections, but you can bring your own handwritten notes. In the computer that you will use, we will provide you a folder with the instructions for the exam, the `ipynb` files to fill with your answers and code, as well some materials of the course.

### Setting up a working environment

Here below we show you one way of setting up a `conda` environment so that you're sure of being able to run all the packages necessary for our class.

1) First of all, you should install `conda ` in your computer as detailed [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
2) Run `conda create -n isla2025` on your terminal.
3) Run `pip install -r requirements.txt` on your terminal.

You should be all set to start working on the TPs. If this is not the case, please make an [issue](https://github.com/ISLA-Grenoble/2025-main/issues) on this repository detailing your problem so that I can either fix it or at least help you out.