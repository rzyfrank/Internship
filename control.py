import sys
import MvImport.CamUse_import1 as c
import threading
import cv2
from queue import Queue


"""
模拟接受plc型号控制拍摄与图像处理的demo
机器信号发出后开始拍照，机器处于待机状态，拍照结束后拍照任务终止等待机器再次发出信号，机器重新开始工作
拍摄的照片单独进行处理，与拍摄过程无关
"""

class machine(threading.Thread):
    def run(self):
        global x
        while True:
            x = input('>>')
            if x == '2':
                with lock_con1:
                    lock_con1.notify()
                print('machine exit')
                break
            elif x == '1':
                print('信号传输')
                with lock_con1:
                    lock_con1.notify()
                print('信号传递结束')
                x = 0
                event1.clear()
                event1.wait()


class take_photo(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        global x
        while True:
            with lock_con1:
                lock_con1.wait()
            if x == '2':
                self.queue.put('end')
                print('take_photo exit')
                break
            else:
                print('开始拍摄')
                img1 = c.access_get_image(cam, nPayloadSize)

                self.queue.put(img1)
                event1.set()
                print('拍摄结束')


class process(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            img = self.queue.get()
            if img == 'end':
                print('process exit')
                break
            else:
                print('开始处理')
                cv2.imshow('img', img)
                cv2.waitKey(0)
                # time.sleep(1)
                self.queue.task_done()
                print('处理结束')


if __name__ == '__main__':
    # if sys.argv[1] == 'Convex':
    #     print('凸透镜模式')
    #     threshold = sys.argv[2]
    #     r = sys.argv[3]
    #     param_a = sys.argv[4]
    #     param_r = sys.argv[5]
    #     scale = sys.argv[6]
    # elif sys.argv[1] == 'Concave':
    #     print('凹透镜模式')
    #     threshold = sys.argv[2]
    lock_con1 = threading.Condition()
    event1 = threading.Event()
    Lock = threading.Lock()
    threads = []
    x = 0
    queue = Queue()
    cam, nPayloadSize = c.startCam()
    threads.append(machine())
    threads.append(take_photo(queue))
    threads.append(process(queue))
    threads[1].start()
    threads[0].start()
    threads[2].start()
    queue.join()
    for t in threads:
        t.join()
    c.stopCam(cam)
