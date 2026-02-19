# Discussion questions: SVMs for binary hate speech detection

Use these to check and deepen understanding of single-label binary classification with SVMs in this demo.

1. **Single vs. multi-label**  
   The dataset has three binary columns (hd, cv, vo), but we collapse them into one binary label (hate / non-hate). Why is that necessary for the SVM setup in this demo? What would change if we kept all three columns as separate targets?

2. **Why an SVM for text?**  
   We use TF-IDF to turn posts into high-dimensional vectors, then a linear SVM. In 1–2 sentences: why is a linear SVM a reasonable choice for such high-dimensional, sparse feature vectors? What might be a drawback?

3. **Margin and support vectors**  
   The SVM fits a separating hyperplane by maximizing the margin between the two classes. In the hate vs. non-hate setting, which training examples become “support vectors”? How might heavy class imbalance (e.g. very few hate examples) affect the learned boundary?

4. **Rebalancing trade-offs**  
   We can balance the training set by oversampling the minority class, undersampling the majority, or using class weights without resampling. For a hate-speech detector, give one advantage and one drawback of each of these three options (e.g. what do we gain or lose in terms of data, runtime, or risk of false positives/negatives)?

5. **Evaluating the classifier**  
   Why is accuracy alone a poor way to evaluate a binary hate-speech model when the test set is imbalanced? What other metric(s) would you use, and how would you explain the trade-off between missing real hate (false negatives) and flagging non-hate as hate (false positives)?
