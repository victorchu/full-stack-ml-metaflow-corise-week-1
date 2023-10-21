from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)
from metaflow.cards import Table, Markdown, Artifact

# TODO move your labeling function from earlier in the notebook here
labeling_function = lambda row: 1 if row['rating'] >= 4 else 0


class BaselineNLPFlow(FlowSpec):
    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter("split-sz", default=0.2)
    
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile("data", default="../data/Womens Clothing E-Commerce Reviews.csv")

    @step
    def start(self):
        # Step-level dependencies are loaded within a Step, instead of loading them
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({"label": labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")



        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        # Get Training and evaluation data
        print(f"- Columns = {self.traindf.columns}")
        text_col = 'review'
        label_col = 'label'
        print("> Getting training data X, y ...")
        X_train = self.traindf[text_col].values
        y_train = self.traindf[label_col].values
        print("> Getting evaluation data X, y ...")
        X_eval = self.valdf[text_col].values
        y_eval = self.valdf[label_col].values
        print(f"- X_train size = {len(X_train)}")
        print(f"- X_eval size = {len(X_eval)}")

        # Create a TF-IDF vectorizer
        print("> Creating a TfidfVectorizer ...")
        vectorizer = TfidfVectorizer()

        # Transform the train and evaluation data into TF-IDF vectors
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_eval_tfidf = vectorizer.transform(X_eval)

        # Create a logistic regression classifier
        classifier = LogisticRegression()

        classifier.fit(X_train_tfidf, y_train)

        # Make predictions on the evaluation data
        y_pred = classifier.predict(X_eval_tfidf)
        y_pred_proba = classifier.predict_proba(X_eval_tfidf)[:, 1]

        ### TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.
        self.base_acc = np.mean(y_pred == y_eval)
        self.base_rocauc = roc_auc_score(y_eval, y_pred_proba)
        print(f"- Accuracy = {self.base_acc}")
        print(f"- ROC = {self.base_rocauc}")

        self.valdf['pred'] = y_pred

        self.next(self.end)

    @card(
        type="corise"
    )  # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):
        import pandas as pd
        pd.options.display.max_colwidth = 400

        msg = "Baseline Accuracy: {}\nBaseline AUC: {}"
        print(msg.format(round(self.base_acc, 3), round(self.base_rocauc, 3)))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        print(f"> append to card: Baseline Accuracy = {self.base_acc}")
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))
        print(f"> Append to card: Baseline ROC = {self.base_rocauc}")
        current.card.append(Markdown("## ORC AUC"))
        current.card.append(Artifact(self.base_rocauc))

        print(f"> Find False Positives ...")
        current.card.append(Markdown("## Examples of False Positives"))
        # TODO: compute the false positive predictions where the baseline is 1 and the valdf label is 0.
        self.fp_df = self.valdf[ (self.valdf.pred==1) & (self.valdf.label==0)]
        print(self.fp_df.head())
        # TODO: display the false_positives dataframe using metaflow.cards
        # Documentation: https://docs.metaflow.org/api/cards#table
        current.card.append(Table.from_dataframe(self.fp_df))

        print(f"> Find False Negatives ...")
        current.card.append(Markdown("## Examples of False Negatives"))
        # TODO: compute the false positive predictions where the baseline is 0 and the valdf label is 1.
        self.fn_df = self.valdf[ (self.valdf.pred==0) & (self.valdf.label==1)]
        print(self.fn_df.head())
        # TODO: display the false_negatives dataframe using metaflow.cards
        current.card.append(Table.from_dataframe(self.fn_df))


if __name__ == "__main__":
    BaselineNLPFlow()
