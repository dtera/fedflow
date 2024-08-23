import os
import tensorflow as tf


class HDFSUtil:
    """Some helper functions for HDFS path.
    """

    def list_sub_dirs(self, path):
        """ List sub dirs for a hdfs path
        """
        subdirs = tf.io.gfile.listdir(path)
        return [os.path.join(path, subdir) for subdir in subdirs]

    def path_exist(self, path):
        return tf.io.gfile.exists(path)

    def is_dir_success(self, path):
        success_file = os.path.join(path, "_SUCCESS")
        return self.path_exist(success_file)

    def list_all_files(self, path, suffix=".gz"):
        files = tf.io.gfile.listdir(path)
        return [os.path.join(path, file) for file in files if file.endswith(suffix)]

    def walk_dir(self, rootdir, depth=2, suffix=".gz"):
        """
        Travese the rootdir to get all files with specified `suffix` within maximum `depth`.
        A directory contain sub directory only or files only,
        and there must exist a file named _SUCCESS for the directory containing files.
        """
        if depth <= 0:
            return []
        if self.is_dir_success(rootdir):
            return self.list_all_files(rootdir, suffix=suffix)
        if depth == 1:
            if self.is_dir_success(rootdir):
                return self.list_all_files(rootdir, suffix=suffix)
            else:
                return []
        res = []
        subdirs = self.list_sub_dirs(rootdir)
        for subdir in subdirs:
            res.extend(self.walk_dir(subdir, depth=depth - 1, suffix=suffix))
        return res

    def download(self, hdfs_path, local_path):
        hadoop_bin = os.path.join(os.getenv("HADOOP_HOME"), "bin", "hadoop")
        cmd = hadoop_bin + " fs -get "
        cmd += " {} {}".format(hdfs_path, local_path)
        ret_code = os.system(cmd)
        if ret_code != 0:
            raise RuntimeError("{} error code: {}".format(cmd, ret_code))

    def upload(self, local_path, hdfs_path):
        hadoop_bin = os.path.join(os.getenv("HADOOP_HOME"), "bin", "hadoop")
        cmd = hadoop_bin + " fs -put -f " + local_path + " " + hdfs_path
        ret_code = os.system(cmd)
        return ret_code == 0

    def touchz(self, path):
        hadoop_bin = os.path.join(os.getenv("HADOOP_HOME"), "bin", "hadoop")
        cmd = hadoop_bin + " fs -touchz " + path
        ret_code = os.system(cmd)
        return ret_code == 0
