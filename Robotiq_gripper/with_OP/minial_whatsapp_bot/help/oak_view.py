#!/usr/bin/env python3
"""OAK-D RGB camera visualization."""
import cv2
import depthai as dai


def main():
    p = dai.Pipeline()

    cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    q = cam.requestOutput((640, 360), dai.ImgFrame.Type.BGR888p).createOutputQueue()

    p.start()
    print("Streaming OAK-D RGB — press 'q' to quit")
    try:
        while p.isRunning():
            cv2.imshow("OAK-D RGB", q.get().getCvFrame())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        p.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
