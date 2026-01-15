from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import bolt_wrench_task
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import load_checkpoint
import os
import numpy as np
import torch
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    # 테스트할 스킬 모델들의 경로 리스트 (폴더 지정 등)
    parser.add_argument('--skill_dir', type=str, default='data/2026-01-15-XXXXX') 
    args = parser.parse_args()

    # Config 로드
    task_path = os.path.dirname(os.path.realpath(__file__))
    cfg = YAML().load(open(task_path + "/rsc/config.yaml", 'r'))
    
    # 시각화 켜기
    cfg['environment']['render'] = True

    # 환경 생성 (Test 모드)
    env = VecEnv(bolt_wrench_task.RaisimGymEnv(task_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
    env.turn_on_visualization()

    # === 스킬 모델 로딩 ===
    # skill_dir 안에 있는 모든 .pt 파일을 로드한다고 가정
    models = []
    model_files = [f for f in os.listdir(args.skill_dir) if f.endswith('.pt') and 'skill' in f]
    model_files.sort() # 순서 정렬
    
    print(f"Loading {len(model_files)} skills...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for f in model_files:
        path = os.path.join(args.skill_dir, f)
        # load_checkpoint는 dict를 반환 (actor, critic, optimizer ...)
        # 여기서는 모델 아키텍처를 복원해야 하므로 PPO 클래스 구조가 필요하거나,
        # 저장된 state_dict를 로드할 빈 모델이 필요함.
        
        # (간소화를 위해 Actor/Critic 네트워크 객체만 리스트에 보관한다고 가정)
        # 실제로는 raisimGymTorch의 load_checkpoint 유틸리티 활용
        checkpoint = load_checkpoint(path)
        
        # 모델 구조 생성 (가중치 로드용)
        # 주의: PPO 클래스 인스턴스를 여러 개 만드는 것은 메모리를 먹으므로
        # Actor/Critic 네트워크만 따로 떼어서 관리하는 것이 효율적.
        # 여기서는 의사 코드로 표현합니다.
        loaded_model = {
            'actor': checkpoint['actor_architecture'].to(device),
            'critic': checkpoint['critic_architecture'].to(device),
            'name': f
        }
        loaded_model['actor'].load_state_dict(checkpoint['actor_state_dict'])
        loaded_model['critic'].load_state_dict(checkpoint['critic_state_dict'])
        loaded_model['actor'].eval()
        loaded_model['critic'].eval()
        
        models.append(loaded_model)

    # === 실행 루프 (Greedy Skill Selection) ===
    
    for i in range(10): # 10번 테스트
        print(f"Test Episode {i+1}")
        env.reset()
        
        # 1.0 (Goal)이 아닌 0.0 (Start)에서 시작하도록 설정
        env.set_curriculum_factor(0.0) 
        
        obs = env.observe()
        
        for step in range(2000):
            best_value = -float('inf')
            best_action = None
            selected_skill_name = ""
            
            obs_tensor = torch.from_numpy(obs).float().to(device)

            # 1. 모든 스킬의 Critic에게 물어봄 (Parallel Evaluation)
            with torch.no_grad():
                for model in models:
                    # Value 평가
                    value = model['critic'](obs_tensor) # Shape: [n_envs, 1]
                    value_scalar = value.item() # Single env 가정
                    
                    if value_scalar > best_value:
                        best_value = value_scalar
                        # Action 추출
                        action_dist = model['actor'](obs_tensor)
                        best_action = action_dist.sample() # or mean
                        selected_skill_name = model['name']
            
            # 2. 가장 높은 Value를 가진 Action 실행
            if best_action is not None:
                # print(f"Step {step}: Selected {selected_skill_name} (V={best_value:.2f})")
                _, _, done, _ = env.step(best_action.cpu().numpy())
                obs = env.observe()
                
                if done.any():
                    print("Episode Done.")
                    break
            else:
                print("No suitable skill found.")
                break
            
            time.sleep(0.01) # 시각화 보기 위해 딜레이

if __name__ == '__main__':
    main()