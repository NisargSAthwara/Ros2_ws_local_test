from setuptools import find_packages, setup

package_name = 'yolo_perception_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/perception_pipeline.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dell_ubuntu',
    maintainer_email='dell_ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'perception_offline = yolo_perception_pkg.perception_offline_node:main',
        'perception_carla = yolo_perception_pkg.perception_carla_node:main',
        'yolo_detector = yolo_perception_pkg.yolo_detector_node:main',
        ],
    },
)
