from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
import numpy as np
from prepare_before_clustring import generate_feature_vec_train, generate_feature_vec_utrain


class ModelRF:
    def __init__(self):
        np.random.seed(1102)
        self.model = RandomForestClassifier(
            n_estimators=25,
            max_depth=4,
            random_state=1102,
            n_jobs=int(np.max([multiprocessing.cpu_count()-2, 1]))
        )

    def fit(self, x, y):
        np.random.seed(1102)
        self.model.fit(x, y)

    def predict(self, x):
        np.random.seed(1102)
        return self.model.predict_proba(x)


def cluster_then_label(Xl, l, Xu):
    np.random.seed(1102)

    Xtot = np.vstack((Xl, Xu))
    print('Xtot', Xtot.shape)
    array_bool_labelled = np.append(np.repeat(True, Xl.shape[0]), np.repeat(False, Xu.shape[0]))
    print('array_bool_labelled', array_bool_labelled.shape)
    ytot = np.append(np.array(l), np.repeat(-1, Xu.shape[0]))
    print('ytot', ytot.shape)

    if Xu.shape[0] > 0:
        scaler = MinMaxScaler()
        Xtot_scaled = scaler.fit_transform(Xtot)
        print('Xtot_scaled', Xtot_scaled.shape)
        # int_opt_n_clusters = get_optimal_n_cluster(Xtot_scaled)
        # initial_centers = X[np.random.choice(range(X.shape[0]), size=int_opt_n_clusters, replace=False), :]

        model_kmeans = KMeans(n_clusters=2439, random_state=1102)
        model_kmeans.fit(Xtot_scaled)
        labels_kmeans = np.array(model_kmeans.labels_)
        print('labels_kmeans', labels_kmeans)
        print(labels_kmeans.shape)

        for k in np.unique(sorted(labels_kmeans)):
            obj_model = ModelRF()
            print(k)
            # print(labels_kmeans == k)
            # print(array_bool_labelled)
            array_bool_l_tmp = array_bool_labelled & (labels_kmeans == k)
            array_bool_u_tmp = (~array_bool_labelled) & (labels_kmeans == k)
            # print(array_bool_u_tmp)
            if (sum(array_bool_l_tmp) > 0) and (sum(array_bool_u_tmp) > 0):

                X_tmp = Xtot[array_bool_l_tmp, :]
                y_tmp = ytot[array_bool_l_tmp]

                if len(np.unique(y_tmp)) == 1:
                    ytot[array_bool_u_tmp] = y_tmp[0]
                else:
                    obj_model.fit(X_tmp, y_tmp)
                    tmp_y_values = sorted(np.unique(y_tmp))
                    # print(obj_model.predict(Xtot[array_bool_u_tmp, :]).argmax(axis=1))
                    ytot[array_bool_u_tmp] = np.take(tmp_y_values,
                                                     obj_model.predict(Xtot[array_bool_u_tmp, :]).argmax(axis=1))

    return Xtot, ytot


def cluster_run(batch_size, model_name, train_path, utrain_path, pl_utrain_path):

    lb_features_v, lb_labels = generate_feature_vec_train(batch_size, train_path, model_name)

    ulb_features_v, ulb_ps, ulabeled_imgs = generate_feature_vec_utrain(batch_size, utrain_path, model_name, pl_utrain_path)

    X_tot, y_tot = cluster_then_label(lb_features_v, lb_labels, ulb_features_v)

    matched_utrain = y_tot[lb_labels.shape[0]:] == ulb_ps

    len(ulabeled_imgs[matched_utrain == True])

    return ulabeled_imgs[matched_utrain == True]

