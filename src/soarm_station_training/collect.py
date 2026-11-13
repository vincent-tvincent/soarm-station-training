"""Native LeRobot demonstration collection; separate from the ROS runtime."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import yaml

class DeviceLock:
    def __init__(self,identity,directory='/tmp/soarm-station-device-locks'):
        Path(directory).mkdir(parents=True,exist_ok=True)
        name=hashlib.sha256(os.path.realpath(str(identity)).encode()).hexdigest()
        self.f=open(Path(directory)/name,'a+')
        try:fcntl.flock(self.f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:self.f.close();raise RuntimeError('Device is owned by production; stop the production service first')
    def close(self):self.f.close()
def native_robot(spec,leader=False):
    import draccus
    if leader:
        from lerobot.teleoperators.so_leader import SO101LeaderConfig,SO100LeaderConfig
        from lerobot.teleoperators.config import TeleoperatorConfig
        from lerobot.teleoperators.utils import make_teleoperator_from_config
        return make_teleoperator_from_config(draccus.decode(TeleoperatorConfig,spec))
    from lerobot.robots.so_follower import SO101FollowerConfig,SO100FollowerConfig
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.utils import make_robot_from_config
    return make_robot_from_config(draccus.decode(RobotConfig,spec))
def collect(config,episodes,seconds,synthetic=False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    c=yaml.safe_load(Path(config).read_text());fps=int(c.get('fps',30));arms=c.get('arms',[]);cameras=c.get('cameras',[])
    ids=[a['id'] for a in arms]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate arm resource ID')
    locks=[];robots={};leaders={};caps={};dataset=None
    if not arms:raise ValueError('Declare at least one arm')
    try:
        if not synthetic:
            # Acquire all physical locks before any connection/configuration.
            identities=[a['robot']['port'] for a in arms]+[a['teleop']['port'] for a in arms]+[str(x['device']) for x in cameras]
            if len(set(os.path.realpath(p) for p in identities))!=len(identities):raise ValueError('Duplicate physical device')
            locks=[DeviceLock(p,c.get('lock_dir','/tmp/soarm-station-device-locks')) for p in identities]
            for a in arms:
                robot=native_robot(a['robot']);leader=native_robot(a['teleop'],True)
                robots[a['id']]=robot;leaders[a['id']]=leader
                robot.connect(calibrate=False);leader.connect(calibrate=False)
            from lerobot.cameras.opencv import OpenCVCamera,OpenCVCameraConfig
            for camera in cameras:
                cap=OpenCVCamera(OpenCVCameraConfig(index_or_path=camera['device'],width=camera.get('width',640),height=camera.get('height',480),fps=fps,fourcc='MJPG'))
                caps[camera['id']]=cap;cap.connect()
        names=[]
        for a in arms:
            names.extend(a['id']+'/'+n for n in (robots[a['id']].action_features if not synthetic else a['features']))
        features={'observation.state':{'dtype':'float32','shape':(len(names),),'names':names},
                  'action':{'dtype':'float32','shape':(len(names),),'names':names}}
        for camera in cameras:features['observation.images.'+camera['id']]={'dtype':'video' if c.get('video',True) else 'image',
            'shape':(camera.get('height',480),camera.get('width',640),3),'names':['height','width','channels']}
        root=Path(c['dataset_root']).expanduser()
        if root.exists():raise FileExistsError('Dataset destination exists; choose a new session path')
        dataset=LeRobotDataset.create(repo_id=c['repo_id'],root=root,robot_type='soarm_station_multi',fps=fps,features=features,
            use_videos=c.get('video',True),image_writer_threads=max(1,len(cameras)*2))
        provenance={'schema_version':1,'synthetic':synthetic,'resource_ids':ids,'native_features':names,'fps':fps,
            'units':c.get('units',{}),'arms':[{k:v for k,v in a.items() if k not in ('robot','teleop')} for a in arms],
            'cameras':[{k:v for k,v in a.items() if k!='device'} for a in cameras]}
        (root/'collection_manifest.json').write_text(json.dumps(provenance,indent=2))
        for episode in range(episodes):
            if not synthetic:input(f'Episode {episode+1}: reset scene and align leaders; press Enter to record ')
            started=time.monotonic();index=0
            while time.monotonic()-started<seconds:
                tick=time.monotonic();state={};action={}
                for a in arms:
                    n=a['id']
                    if synthetic:
                        values={k:float(np.sin(index/fps)*.1) for k in a['features']};sent=values
                    else:
                        values=robots[n].get_observation();sent=leaders[n].get_action();robots[n].send_action(sent)
                    for k in (a['features'] if synthetic else robots[n].action_features):state[n+'/'+k]=values[k];action[n+'/'+k]=sent[k]
                frame={'observation.state':np.array([state[n] for n in names],dtype=np.float32),
                       'action':np.array([action[n] for n in names],dtype=np.float32),'task':c['task']}
                for camera in cameras:
                    if synthetic:
                        image=np.zeros((camera.get('height',480),camera.get('width',640),3),dtype=np.uint8);image[:,:,episode%3]=100+index%100
                    else:image=caps[camera['id']].async_read(timeout_ms=500)
                    frame['observation.images.'+camera['id']]=image
                dataset.add_frame(frame);index+=1;time.sleep(max(0,1/fps-(time.monotonic()-tick)))
            dataset.save_episode();print(f'Saved episode {episode+1}: {index} frames')
        dataset.finalize();dataset=None
        return str(root)
    finally:
        if dataset is not None:dataset.finalize()
        for robot in robots.values():
            # The configured native driver owns its documented hold/disconnect policy.
            if robot.is_connected:robot.disconnect()
        for leader in leaders.values():
            if leader.is_connected:leader.disconnect()
        for cap in caps.values():
            if cap.is_connected:cap.disconnect()
        for lock in locks:lock.close()
