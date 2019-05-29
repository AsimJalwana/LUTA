import sqlite3
import sys

import numpy as np

is_python3 = sys.version_info[0] >= 3
print("sqlite version = ", sqlite3.sqlite_version)

create_attack_info_table = """create table if not exists attackinfo (
           attackid         integer primary key autoincrement not null,
           description      text    not null,
           classifierid     integer not null,
           foreign key(classifierid) references classifierinfo(classifierid) on delete cascade
           )"""

create_wnid_info_table = """create table if not exists wnidinfo (
           wnid             text    primary key not null,
           description      text    not null
           )"""

create_image_info_table = """create table if not exists imageinfo (
       imageid          integer primary key autoincrement not null,
       description      text    not null,
       wnid             integer not null,
       resolution       text, 
       foreign key(wnid) references wnidinfo(wnid) 
       )"""

create_classifier_info_table = """create table if not exists classifierinfo (
       classifierid     integer primary key not null,
       resolution       integer not null, 
       description      text    not null
       )"""

create_classifier_image_table = """create table if not exists classifierimage (
       classifierid     integer not null,
       imageid          integer not null,
       plabel           integer not null,
       primary key (classifierid, imageid),
       foreign key(classifierid) references classifierinfo(classifierid) on delete cascade,
       foreign key(imageid)      references imageinfo(imageid)           on delete cascade
       )"""

create_attackalgorithm_info_table = """create table if not exists attackalgorithminfo (
       algorithmid           integer primary key not null,
       description           text not null
       )"""

create_hyperparameter_info_table = """create table if not exists hyperparameterinfo (
       attackid              integer primary key not null,
       eta                   real not null,
       beta1                 real not null,
       beta2                 real not null,
       batchsize             integer not null, 
       algorithmid           integer not null,
       groundlabel           integer not null,
       targetlabel           integer not null,
       OtherParameters       text, 
       foreign key(attackid)    references attackinfo(attackid)             on delete cascade,
       foreign key(algorithmid) references attackalgorithminfo(algorithmid) on delete cascade
       )"""

create_attack_table = """create table if not exists attack (
       attackid               integer not null,
       iteration              integer not null,
       epoch                  integer not null,
       perturbedImage         blob,
       upsilonImage           blob,
       omegaImage             blob,
       primary key(attackid, iteration),
       foreign key(attackid)         references attackinfo(attackid)                 on delete cascade
       )"""

create_attack_training_performance_table = """create table if not exists attacktrainingperformance (
       attackid               integer not null,
       iteration              integer not null,
       successRatio           integer not null,
       primary key(attackid, iteration),
       foreign key(attackid)         references attackinfo(attackid)                 on delete cascade
       )"""

create_attack_testing_performance_table = """create table if not exists attacktestingperformance (
       attackid               integer not null,
       iteration              integer not null,
       successRatio           integer not null,
       primary key(attackid, iteration),
       foreign key(attackid)         references attackinfo(attackid)                 on delete cascade
       )"""

create_trainingtesting_images_table = """create table if not exists trainingtestingimages (
       attackid               integer not null,
       imageid                integer not null,
       istraining             integer not null,
       primary key(attackid, imageid),
       foreign key(imageid)   references imageinfo(imageid) on delete cascade
       )"""

create_attack_training_prediction_table = """create table if not exists attacktrainingprediction (
       attackid               integer not null,
       iteration              integer not null,
       topprediction          integer not null,
       primary key(attackid, iteration),
       foreign key(attackid)         references attackinfo(attackid)                 on delete cascade
       )"""

create_attack_training_all_predictions_table = """create table if not exists attacktrainingallpredictions (
       attackid               integer not null,
       iteration              integer not null,
       allprediction          blob,
       primary key(attackid, iteration),
       foreign key(attackid)         references attackinfo(attackid)                 on delete cascade
       )"""

create_attack_testing_prediction_table = """create table if not exists attacktestingprediction (
       attackid               integer not null,
       iteration              integer not null,
       topprediction          integer not null,
       primary key(attackid, iteration),
       foreign key(attackid)         references attackinfo(attackid)                 on delete cascade
       )"""

create_attack_testing_all_predictions_table = """create table if not exists attacktestingallpredictions (
       attackid               integer not null,
       iteration              integer not null,
       allprediction          blob,
       primary key(attackid, iteration),
       foreign key(attackid)         references attackinfo(attackid)                 on delete cascade
       )"""

create_all = "; ".join([
    create_classifier_info_table,
    create_attack_info_table,
    create_wnid_info_table,
    create_image_info_table,
    create_classifier_image_table,
    create_attackalgorithm_info_table,
    create_trainingtesting_images_table,
    create_attack_training_performance_table,
    create_attack_testing_performance_table,
    create_attack_table,
    create_hyperparameter_info_table,
    create_attack_training_prediction_table,
    create_attack_testing_prediction_table,
    create_attack_training_all_predictions_table,
    create_attack_testing_all_predictions_table
])


def array_to_blob(array):
    if is_python3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if is_python3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class GradientBasedAttacksDatabase(sqlite3.Connection):
    logger = None

    @staticmethod
    def connect(database_path, logger):
        GradientBasedAttacksDatabase.logger = logger
        return sqlite3.connect(database_path, factory=GradientBasedAttacksDatabase)

    def __init__(self, *args, **kwargs):
        super(GradientBasedAttacksDatabase, self).__init__(*args, **kwargs)
        self.create_tables = lambda: self.executescript(create_all)

    def get_attackinfo(self, attackid=None):
        cursor = self.execute("select attackid, description, classifierid from attackinfo where attackid = " + str(attackid))
        row = cursor.fetchone()

        if row is not None:
            return row[0], row[1], row[2]
        else:
            return None, None, None

    def add_attackInfo(self, description, classifierid, attackid=None):
        GradientBasedAttacksDatabase.logger.debug("Entry has been added to U:AttackInfo for classifierID= " + str(
            classifierid) + " with description = " + description)
        cursor = self.execute("insert into attackinfo values (?, ?, ? )", (attackid, description, classifierid))
        return cursor.lastrowid

    def add_wnidInfo(self, wnid, description):
        cur = self.execute("select * from wnidinfo where wnid=(?)", (wnid,))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry has been added to U:WnIdInfo for wnidinfo = " + str(wnid) + " with description = " + description)
            cursor = self.execute("insert into wnidinfo values (?, ?)", (wnid, description))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug("Entry found in U:WnIdInfo for wnidinfo = " + str(wnid))

    def doesImageInfoExists(self, description, wnid):
        cur = self.execute("select imageId, description from imageinfo where description=? and wnid=?",
                           (description, wnid))
        entry = cur.fetchone()

        if entry is None:
            return None, None
        else:
            return entry[0], entry[1]

    def add_imageInfo(self, description, wnid, resolution, imageid=None):
        id, _ = self.doesImageInfoExists(description, wnid)
        if id is None:
            GradientBasedAttacksDatabase.logger.debug("Image with Description = " + description + " and WnID = " + str(
                wnid) + " has been inserted into U:imageinfo.")
            cursor = self.execute("insert into imageinfo values (?, ?, ?, ?)",
                                  (imageid, description, wnid, resolution))
            return cursor.lastrowid
        else:
            return id

    def add_classifierImage(self, classifierid, imageid, plabel):
        cur = self.execute("select imageId from classifierimage where classifierid=? and imageid=?",
                           (classifierid, imageid))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry added in U:classifierImage for classifierID = " + str(classifierid) + " and ImageID = " + str(
                    imageid))
            cursor = self.execute("insert into classifierimage values (?, ?, ?)", (classifierid, imageid, plabel))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry found in U:classifierImage for classifierID = " + str(classifierid) + " and ImageID = " + str(
                    imageid))

    def add_classifierInfo(self, classifierid, resolution, description):
        cur = self.execute("select * from classifierinfo where classifierid=(?)", (classifierid,))
        entry = cur.fetchone()

        if entry is None:
            cursor = self.execute("insert into classifierinfo values (?, ?, ?)",
                                  (classifierid, resolution, description))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry found in U:ClassifierInfo for classifierid = " + str(classifierid))

    def add_attackalgorithmInfo(self, algorithmid, description):
        cur = self.execute("select * from attackalgorithminfo where algorithmid=(?)", (algorithmid,))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug("Entry added in U:GradientMethodId for AlgorithmId = " + str(
                algorithmid) + " and description = " + description)
            cursor = self.execute("insert into attackalgorithminfo values (?, ?)", (algorithmid, description))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry found in U:GradientMethodId for AlgorithmId = " + str(algorithmid))

    def get_hyperparameterinfo(self, attackid):
        cur = self.execute("select attackid, eta, beta1, beta2, batchsize, algorithmid, groundlabel, targetlabel from hyperparameterinfo where attackid=" + str(attackid))
        row = cur.fetchone()

        if row[0] is None:
            return None, None, None, None, None, None, None, None
        else:
            return row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]

    def add_hyperparameterInfo(self, attackid, eta, beta1, beta2, batchsize, algorithmid, groundlabel, targetlabel,
                                    otherParameters=''):
        cur = self.execute("select * from hyperparameterinfo where attackid=?", (attackid,))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry Added in U:HyperparameterInfo for AttackId = " + str(attackid))
            cursor = self.execute("insert into hyperparameterinfo values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                  (attackid, eta, beta1, beta2, batchsize, algorithmid, groundlabel, targetlabel,
                                   otherParameters))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry found in U:HyperparameterInfo for AttackId = " + str(attackid))

    def add_attack(self, attackid, iteration, epoch, perturbedImage, upsilonImage, omegaImage):
        cur = self.execute("select * from attack where attackid = ? and iteration = ? ", (attackid, iteration))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry in U:attack for " + str(attackid) + " and iteration = " + str(
                    iteration) + " has been added.")
            cursor = self.execute("insert into attack values (?, ?, ?, ?, ?, ?)",
                                  (attackid, iteration, epoch, array_to_blob(perturbedImage),
                                   array_to_blob(upsilonImage), array_to_blob(omegaImage)))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "There is an entry in U:attack for " + str(attackid) + " and iteration = " + str(
                    iteration) + " .... Skipping")

    def add_attack_training_performance(self, attackid, iteration, successratio):
        cur = self.execute("select * from attacktrainingperformance where attackid = ? and iteration = ? ",
                           (attackid, iteration))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry in U:attacktrainingperformance for " + str(attackid) + " and iteration = " + str(
                    iteration) + " has been added.")
            cursor = self.execute("insert into attacktrainingperformance values (?, ?, ?)",
                                  (attackid, iteration, successratio))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "There is an entry in U:attacktrainingperformance for " + str(attackid) + " and iteration = " + str(
                    iteration) + ".... Skipping")

    def add_attack_testing_performance(self, attackid, iteration, successratio):
        cur = self.execute("select * from attacktestingperformance where attackid = ? and iteration = ? ",
                           (attackid, iteration))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry in U:attacktestingperformance for " + str(attackid) + " and iteration = " + str(
                    iteration) + " has been added.")
            cursor = self.execute("insert into attacktestingperformance values (?, ?, ?)",
                                  (attackid, iteration, successratio))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "There is an entry in U:attacktrainingperformance for " + str(attackid) + " and iteration = " + str(
                    iteration) + ".... Skipping")

    def add_trainingtesting_images(self, attackid, imageid, istraining):
        cur = self.execute("select * from trainingtestingimages where attackid = ? and imageid = ?",
                           (attackid, imageid))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry in U:trainingtestingimages for attackid = " + str(attackid) + ", imageid = " + str(
                    imageid) + ", istraining = " + str(istraining) + " has been added.")
            cursor = self.execute("insert into trainingtestingimages values (?, ?, ?)", (attackid, imageid, istraining))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "There is an U:entry in trainingtestingimages for attackid = " + str(attackid) + ", imageid = " + str(
                    imageid) + ", istraining = " + str(istraining) + " .... Skipping")

    def loadAllTrainingPredictions(self, attackid):
        query = """SELECT iteration, allprediction 
                     FROM AttackTrainingAllPredictions
                    WHERE attackid = """ + str(attackid)

        cur = self.execute(query)
        rows = cur.fetchall()

        histogram = np.zeros((len(rows), 1000))

        print(len(rows))
        for row in rows:
            print (row[0], blob_to_array(row[1], dtype=float, shape=(1,1000)))
            histogram[row[0],:] = blob_to_array(row[1], dtype=float, shape=(1,1000))

        return histogram


    def findDataFromLastRun(self, attackid, targetSize=(224, 224)):
        query = """SELECT a.attackid, a.iteration, a.epoch, a.perturbedimage, a.upsilonimage, a.omegaimage
                     FROM attack a
                    INNER JOIN (
                    SELECT attackid, MAX(epoch) epoch, MAX(iteration) iteration
                      FROM attack
                     GROUP BY attackid) b ON a.attackid = b.attackid AND a.epoch = b.epoch AND a.iteration = b.iteration AND a.attackid = """ + str(
            attackid)

        cur = self.execute(query)
        row = cur.fetchone()

        if row is None:
            GradientBasedAttacksDatabase.logger.debug(
                "No Saved entry found in U:attack for attackid = " + str(attackid) + " in U:Attack")
            return None, None, None, None, None, None
        else:
            return row[0], row[1], row[2], blob_to_array(row[3], np.float32, shape=(targetSize[0], targetSize[1], -1)), blob_to_array(
                row[4], np.float32, shape=(targetSize[0], targetSize[1], -1)), blob_to_array(row[5], np.float32, shape=(targetSize[0], targetSize[1], -1))

    def add_attack_training_prediction(self, attackid, iteration, prediction):
        cur = self.execute("select * from attacktrainingprediction where attackid = ? and iteration = ? ",
                           (attackid, iteration))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry in U:attacktrainingprediction for " + str(attackid) + " and iteration = " + str(
                    iteration) + " has been added.")
            cursor = self.execute("insert into attacktrainingprediction values (?, ?, ?)",
                                  (attackid, iteration, prediction))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "There is an entry in U:attacktrainingprediction for " + str(attackid) + " and iteration = " + str(
                    iteration) + ".... Skipping")

    def add_attack_training_all_predictions(self, attackid, iteration, predictions):
        cur = self.execute("select * from attacktrainingallpredictions where attackid = ? and iteration = ? ",
                           (attackid, iteration))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry in U:attacktrainingallpredictions for " + str(attackid) + " and iteration = " + str(
                    iteration) + " has been added.")
            cursor = self.execute("insert into attacktrainingallpredictions values (?, ?, ?)",
                                  (attackid, iteration, array_to_blob(predictions)))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "There is an entry in U:attacktrainingallpredictions for " + str(attackid) + " and iteration = " + str(
                    iteration) + ".... Skipping")

    def add_attack_testing_prediction(self, attackid, iteration, prediction):
        cur = self.execute("select * from attacktestingprediction where attackid = ? and iteration = ? ",
                           (attackid, iteration))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry in U:attacktestingprediction for " + str(attackid) + " and iteration = " + str(
                    iteration) + " has been added.")
            cursor = self.execute("insert into attacktestingprediction values (?, ?, ?)",
                                  (attackid, iteration, prediction))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "There is an entry in U:attacktestingprediction for " + str(attackid) + " and iteration = " + str(
                    iteration) + ".... Skipping")

    def add_attack_testing_all_predictions(self, attackid, iteration, prediction):
        cur = self.execute("select * from attacktestingallpredictions where attackid = ? and iteration = ? ",
                           (attackid, iteration))
        entry = cur.fetchone()

        if entry is None:
            GradientBasedAttacksDatabase.logger.debug(
                "Entry in U:attacktestingallpredictions for " + str(attackid) + " and iteration = " + str(
                    iteration) + " has been added.")
            cursor = self.execute("insert into attacktestingallpredictions values (?, ?, ?)",
                                  (attackid, iteration, array_to_blob(prediction)))
            self.commit()
        else:
            GradientBasedAttacksDatabase.logger.debug(
                "There is an entry in U:attacktestingallpredictions for " + str(attackid) + " and iteration = " + str(
                    iteration) + ".... Skipping")


    def clearAttackTestingTrainingPerformance(self, attackid, iterationStart):
        GradientBasedAttacksDatabase.logger.warning(
            "Clearing the U:attacktrainingperformance including iteration " + str(iterationStart))
        query = "delete from attacktrainingperformance where iteration >= " + str(iterationStart) + " and attackid = " + str(attackid)
        cur = self.execute(query)

        query = "delete from attacktestingperformance where iteration >= " + str(iterationStart) + " and attackid = " + str(attackid)
        cur = self.execute(query)

        query = "delete from attacktrainingprediction where iteration >= " + str(iterationStart) + " and attackid = " + str(attackid)
        cur = self.execute(query)

        query = "delete from attacktestingprediction where iteration >= " + str(iterationStart) + " and attackid = " + str(attackid)
        cur = self.execute(query)

        query = "delete from attacktrainingallpredictions where iteration >= " + str(iterationStart) + " and attackid = " + str(attackid)
        cur = self.execute(query)

        query = "delete from attacktestingallpredictions where iteration >= " + str(iterationStart) + " and attackid = " + str(attackid)
        cur = self.execute(query)

        self.commit()

    def loadHyperparametersForTheAttack(self, attackid):
        query = " select eta, beta1, beta2, batchsize, algorithmid, groundlabel, targetlabel from hyperparameterinfo where attackid = " + str(
            attackid)
        cur = self.execute(query)
        row = cur.fetchone()

        if row is None:
            GradientBasedAttacksDatabase.logger.error("Hyper parameters not found! for attackid = " + str(attackid))
            return None, None, None, None, None, None, None
        else:
            return row[0], row[1], row[2], row[3], row[4], row[5], row[6]

    def readFromImageInfo(self, imageId, targetSize=None):
        cur = self.execute("select image, resolution from imageinfo where imageid = (?)", (imageId,))
        rows = cur.fetchall()

        for row in rows:
            if targetSize == None:
                resolutionStr = row[1]
                elementsStr = resolutionStr.split(",")
                elementsInt = [int(i) for i in elementsStr]
                imageArray = blob_to_array(row[0], np.float32, shape=(elementsInt[0], elementsInt[1], elementsInt[2]))
            else:
                imageArray = blob_to_array(row[0], np.float32, shape=(targetSize[0], targetSize[1], -1))

        return imageArray



def example_usage():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("error: database path already exists -- will not modify it.")

    # open the database.
    db = GradientBasedAttacksDatabase.connect(args.database_path)

    # lets enable the foreign keys
    print("enabling fkeys ", db.execute("pragma foreign_keys = on"))
    db.create_tables()


if __name__ == "__main__":
    example_usage()
