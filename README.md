# Project 4: Machine Learning Fairness

### [Project Description](doc/project4_desc.md)

Term: Spring 2023

+ Team 6
+ Projec title: Machine Learning Fairness
+ Team members
	+ Tianxiao He
	+ Linda Lin
	+ Xinming Pan
	+ Namira Suniaprita
	+ Han Wang
	+ Yixun Xu
+ Project summary: In this project, we investigated fairness in machine learning algorithm using [Compas dataset](https://github.com/propublica/compas-analysis/). This dataset contains criminal history for defendants from Broward County and we aim to predict 2 year recidivism of each defendent without introducing bias on sensitive attributes such as race. Specifically, two research papers were examined [Fairness Constraints: Mechanisms for Fair Classification](https://arxiv.org/pdf/1507.05259.pdf) (A2) and [Information Theoretic Measures for Fairness-aware Feature Selection](https://arxiv.org/pdf/2106.00772.pdf) (A7). The data was cleaned and preprocessed to handle categorical values. For paper A2, we implemented baseline and fairness constrained logistic regression as well as baseline and fairness constrained support vector machine. For paper A7, we quantified the discrimination score of each feature by computing Shapley coefficients and the features were selected using FFS algorithm. Logistic regression and support vector machine models were then implemented using these pre-selected features. As a result, the FFS models with logistic regression were found to outperform other models, with accuracy 61.80% and P-rule 65.54%. 
	

**Contribution statement**: [default] All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 

Xinming Pan, Yixun Xu, and Han Wang works on algorithms from paper A7. Namira Suniaprita, Linda Lin, and Tianxiao He works on algorithms from paper A2. Namira Suniaprita preprocessed the data and implemented the constrained logistic regression. Linda Lin implemented the constrained SVM. Tianxiao He combined the code in main.ipynb, evaluated all model's performance and performed data visualization. 

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
