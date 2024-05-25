from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat


class Agent(Node):
    def __init__(self):
        super().__init__(
            "agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.agent_id = self.get_parameter("id").value
        self.neighbors = self.get_parameter("Nii").value
        self.x_i = self.get_parameter("xzero").value
        self.get_logger().info(f"I am agent: {self.agent_id}")

        self.t = 0

        for j in self.neighbors:
            print(self.neighbors)
            self.create_subscription(
                MsgFloat, f"/topic_{j}", self.listener_callback, 10
            )

        self.publisher = self.create_publisher(
            MsgFloat, f"/topic_{self.agent_id}", 10
        )  # topic_i

        timer_period = 2
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.received_data = {j: [] for j in self.neighbors}

    def listener_callback(self, msg):
        j = int(msg.data[0])
        msg_j = list(msg.data[1:])
        self.received_data[j].append(msg_j)

        return None

    def timer_callback(self):
        msg = MsgFloat()

        if self.t == 0:
            msg.data = [float(self.agent_id), float(self.t), float(self.x_i)]
            self.publisher.publish(msg)
            self.get_logger().info(f"Iter: {self.t} x_{self.agent_id}: {self.x_i}")

            self.t += 1
        else:
            all_received = all(
                self.t - 1 == self.received_data[j][0][0] for j in self.neighbors
            )
            if all_received:
                local_max = self.x_i
                for j in self.neighbors:
                    _, x_j = self.received_data[j].pop(0)
                    local_max = max(local_max, x_j)

                self.x_i = local_max  # update the local variable

                msg.data = [float(self.agent_id), float(self.t), float(self.x_i)]
                self.publisher.publish(msg)
                self.get_logger().info(f"Iter: {self.t} x_{self.agent_id}: {self.x_i}")

                self.t += 1


def main():
    rclpy.init()

    agent = Agent()
    sleep(1)
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
