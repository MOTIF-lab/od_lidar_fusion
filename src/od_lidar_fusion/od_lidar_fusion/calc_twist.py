import rclpy
from rclpy.node import Node
from fusion_msgs.msg import FusionDetections, FusionDetection


class TwistCalculation(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('MyNode has been started.')

def main(args=None):
    rclpy.init(args=args)
    my_node = MyNode()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()