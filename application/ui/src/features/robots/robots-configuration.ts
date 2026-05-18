import { SchemaRobot, SchemaRobotType } from './robot-types';

export const isFollower = (robot: SchemaRobot) => {
    return robot.type === 'SO101_Follower' || robot.type === 'Trossen_WidowXAI_Follower';
};

export const isLeader = (robot: SchemaRobot) => {
    return robot.type === 'SO101_Leader' || robot.type === 'Trossen_WidowXAI_Leader';
};

/** Resolve a `SchemaRobotType` to its URDF asset path. */
export const urdfPathForType = (robotType: SchemaRobotType): string => {
    if (robotType !== undefined && robotType.toLowerCase().includes('trossen')) {
        return '/widowx/urdf/generated/wxai/wxai_follower.urdf';
    }
    return '/SO101/so101_new_calib.urdf';
};

const SO101_TO_URDF = {
    'shoulder_pan.pos': ['shoulder_pan'],
    'shoulder_lift.pos': ['shoulder_lift'],
    'elbow_flex.pos': ['elbow_flex'],
    'wrist_flex.pos': ['wrist_flex'],
    'wrist_roll.pos': ['wrist_roll'],
    'gripper.pos': ['gripper'],
};
const TROSSEN_TO_URDF = {
    'shoulder_pan.pos': ['joint_0'],
    'shoulder_lift.pos': ['joint_1'],
    'elbow_flex.pos': ['joint_2'],
    'wrist_flex.pos': ['joint_3'],
    'wrist_yaw.pos': ['joint_4'],
    'wrist_roll.pos': ['joint_5'],
    'gripper.pos': ['left_carriage_joint', 'right_carriage_joint'],
};

export const ROBOT_TYPE_TO_URDF_MAP: Record<SchemaRobotType, Record<string, string[]>> = {
    SO101_Follower: SO101_TO_URDF,
    SO101_Leader: SO101_TO_URDF,
    Trossen_WidowXAI_Leader: TROSSEN_TO_URDF,
    Trossen_WidowXAI_Follower: TROSSEN_TO_URDF,
};
