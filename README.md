# Project 4: Machine Learning Fairness

<img src="figs/polarization.jpg" width="600">
Photo source: https://www.uva.nl/en/shared-content/faculteiten/en/faculteit-der-maatschappij-en-gedragswetenschappen/news/2021/05/we-are-not-as-polarized-as-we-think.html?cb

### [Project Description](doc/project4_desc.md)

Term: Spring 2023

+ **Team 6**
+ **Project title:** Machine Learning Fairness : Optimizing Accuracy While Adhering to Fairness Constraints (C-LR and C-SVM) and Information Theoretic Measures for Fairness-Aware Feature Selection (FFS)
+ **Team members**
	+ [Tianxiao He](yl5144@columbia.edu) (yl5144@columbia.edu)
	+ [Linda Lin](yl5144@columbia.edu) (yl5144@columbia.edu)
	+ [Xinming Pan](xp2203@columbia.edu) (xp2203@columbia.edu)
	+ [Namira Suniaprita](https://www.linkedin.com/in/namira-suniaprita-b32372125/) (ns3646@columbia.edu)  
	+ [Han Wang](hw2900@columbia.edu) (hw2900@columbia.edu)
	+ [Yixun Xu](yx2740@columbia.edu) (yx2740@columbia.edu)

+ **Project summary:** In this project, we investigated fairness in machine learning algorithm using [Compas dataset](https://github.com/propublica/compas-analysis/). This dataset contains criminal history for defendants from Broward County and we aim to predict 2 year recidivism of each defendent without introducing bias on sensitive attributes such as race. Specifically, two research papers were examined [Fairness Constraints: Mechanisms for Fair Classification](https://arxiv.org/pdf/1507.05259.pdf) (A2) and [Information Theoretic Measures for Fairness-aware Feature Selection](https://arxiv.org/pdf/2106.00772.pdf) (A7). The data was cleaned and preprocessed to handle categorical values. For paper A2, we implemented baseline and fairness constrained logistic regression as well as baseline and fairness constrained support vector machine. For paper A7, we quantified the discrimination score of each feature by computing Shapley coefficients and the features were selected using FFS algorithm. Logistic regression and support vector machine models were then implemented using these pre-selected features. As a result, the FFS models with logistic regression were found to outperform other models, with accuracy 61.80% and P-rule 65.54%. 

+ **Outcomes summary**: Taking both accuracy and fairness into consideration, we recommend FFS-LR, which has 61.8% accuracy as the second highest among all models, and a p-rule score of 65.54% that performs reasonably well (ranking third among all models). We don't choose C-LR or C-SVM because, even though they have high P-rule values, their accuracy is too low for a binary classifier. We also don't choose traditional LR or SVM because, despite their relatively high accuracy, they do not account for fairness.

  <img src="figs/results_table.png" width="400">
  <img src="figs/results.png" width="500">

+ **Employed technologies**: Python was employed for model development, including logistic regression (LR), support vector machines (SVM), constrained logistic regression (C-LR), constrained support vector machines (C-SVM), and fairness-aware feature selection (FFS).

**Contribution statement**: [default] All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 

Xinming Pan, Yixun Xu, and Han Wang works on algorithms from paper A7. Namira Suniaprita, Linda Lin, and Tianxiao He works on algorithms from paper A2. Namira Suniaprita preprocessed the data, implemented the constrained logistic regression (C-LR), and tuning the hyperparameters of fairness and accuracy (fairness, accuracy, separation, and gamma). Linda Lin implemented the constrained SVM. Tianxiao He combined the code in main.ipynb, evaluated all model's performance and performed data visualization. Han Wang wrote the structure of main.ipynb, preprocessed the data, implemented the baseline models, and did the presentation. Xinming Pan worked on the A7 algorithm and finalized it, then he build the SVM and Logistic Regression by using FFS Algorithm. YIXun Xu implemented the A7 algorithm.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
