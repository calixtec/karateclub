from karateclub.dataset import GraphReader
from karateclub import Diff2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


reader = GraphReader("twitch")


graph = reader.get_graph()
y = reader.get_target()




model = Diff2Vec(diffusion_number=2, diffusion_cover=20, dimensions=16)
model.fit(graph)
X = model.get_embedding()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_hat = downstream_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_hat)
print('AUC: {:.4f}'.format(auc))
graph_reader = GraphReader(dataset="wikipedia")
#graph_reader.display_graph()
