# Bakeoff!

I haven't come across many blog posts comparing "standard" approaches to text classification.
In this (soon-to-be) notebook, I'm going to evaluate logistic regression (a standard industry workhorse)
against a ConvNet (the "new kid" baseline) on Twitter sentiment analysis.

(This work was inspired by some experiments I was doing on search query classification with some data that I can't share.)

## The contenders

1. scikit-learn pipeline with a char-based CountVectorizer (3,4,5,6-grams) and LogisticRegression (with xval'd regularizer)
2. basic ConvNet adhering closely to the architecture in Kim (2014) with filters of size 3,4,5,6 (100 each)

## Eval

One of my pet peeves about posts like this is the lack of detail w.r.t. things like system hardware/architecture,
training time, etc. as well as the simplicity of the eval metrics (usually just raw accuracy).
