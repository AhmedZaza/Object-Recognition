import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from glob import glob

class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []
        self.sift_object = cv2.xfeatures2d.SIFT_create()
        self.kmeans_obj = KMeans(n_clusters=no_clusters)
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.clf = SVC(C=.1, kernel='linear')

    def cluster(self):
        """
        cluster using KMeans algorithm,

        """
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

    def developVocabulary(self, n_images, descriptor_list, kmeans_ret=None):
        self.mega_histogram = np.array([np.zeros(self.no_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            l = len(descriptor_list[i])
            for j in range(l):
                if kmeans_ret is None:
                    idx = self.kmeans_ret[old_count + j]
                else:
                    idx = kmeans_ret[old_count + j]
                self.mega_histogram[i][idx] += 1
            old_count += l
        print("Vocabulary Histogram Generated")

    def standardize(self, std=None):
        """

        standardize is required to normalize the distribution
        wrt sample size and features. If not normalized, the classifier may become
        biased due to steep variances.

        """
        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)
            self.mega_histogram = self.scale.transform(self.mega_histogram)
        else:
            print("STD not none. External STD supplied")
            self.mega_histogram = std.transform(self.mega_histogram)

    def formatND(self, l):
        """
        restructures list into vstack array of shape
        M samples x N features for sklearn

        """
        vStack = np.array(l[0])
        for remaining in l:
            vStack = np.vstack((vStack, remaining))
        self.descriptor_vstack = vStack.copy()
        return vStack

    def train(self, train_labels):
        self.clf.fit(self.mega_histogram, train_labels)
        print("Training completed")
        final_data = np.column_stack((self.mega_histogram, train_labels))
        np.savetxt("Sift_Training.csv", final_data, delimiter=',', fmt='%f')

    def predict(self, iplist):
        predictions = self.clf.predict(iplist)
        return predictions

    def plotHist(self, vocabulary=None):
        print("Plotting histogram")
        if vocabulary is None:
            vocabulary = self.mega_histogram

        x_scalar = np.arange(self.n_clusters)
        y_scalar = np.array([abs(np.sum(vocabulary[:, h], dtype=np.int32)) for h in range(self.n_clusters)])

        print(y_scalar)

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()


    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]

    def getFiles(self, path):
        imlist = {}
        count = 0
        for each in glob(path + "*"):
            word = each.split("/")[-1]
            imlist[word] = []
            for imagefile in glob(path + word + "/*"):
                im = cv2.imread(imagefile, 0)
                imlist[word].append(im)
                count += 1
        return imlist, count

    def trainModel(self):
        # read file. prepare file lists.
        self.images, self.trainImageCount = self.getFiles(self.train_path)
        # extract SIFT Features from each image
        label_count = 0
        count = 0
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            for im in imlist:
                self.train_labels = np.append(self.train_labels, label_count)
                kp, des = self.features(im)
                self.descriptor_list.append(des)
            label_count += 1
        # perform clustering
        self.formatND(self.descriptor_list)
        self.cluster()
        self.developVocabulary(n_images=self.trainImageCount, descriptor_list=self.descriptor_list)
        self.standardize()
        self.train(self.train_labels)

    def recognize(self, test_img, test_image_path=None):
        kp, des = self.features(test_img)

        # generate vocab for test image
        vocab = np.array([0 for i in range(self.no_clusters)])
        # locate nearest clusters for each of
        # the visual word (feature) present in the image

        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.kmeans_obj.predict(des)

        for each in test_ret:
            vocab[each] += 1

        # Scale the features
        vocab = self.scale.transform([vocab])

        # predict the class of the image
        lb = self.clf.predict(vocab)
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb

    def testModel(self):
        """
        This method is to test the trained classifier

        read all images from testing path
        use BOVHelpers.predict() function to obtain classes of each image

        """

        self.testImages, self.testImageCount = self.getFiles(self.test_path)

        predictions = []
        correct = 0.0
        all = 0.0
        for word, imlist in self.testImages.items():
            for im in imlist:
                cl = self.recognize(im)
                predictions.append({
                    'image': im,
                    'class': cl,
                    'object_name': self.name_dict[str(int(cl[0]))]
                })
                if self.name_dict[str(int(cl[0]))] == word:
                    correct += 1.0
                all += 1.0;

       # print("Success With Accuracy = ", (correct/all)*100.0)
        #print(predictions)
        for each in predictions:
            plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
            plt.title(each['object_name'])
            plt.show()
        return correct / all

    def print_vars(self):
        pass


bov = BOV(100)
bov.test_path = "/Users/mac/PycharmProjects/PR/images/test/"
bov.train_path = "/Users/mac/PycharmProjects/PR/images/train/"
bov.trainModel()
print(bov.testModel())