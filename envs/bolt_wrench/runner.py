from ruamel.yaml import YAML
import io
import sys
import os
# 현재 파일 위치 기준 'build' 폴더 경로 추가
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir + "/build")
# 빌드된 라이브러리 이름(libraisim_gym_env.so)을 import
# (주의: CMakeLists.txt에서 add_library 이름을 'raisim_gym_env'로 했다면 아래처럼 씁니다)
# import raisim_gym_env as bolt_wrench
import bolt_wrench
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.bolt_wrench import NormalSampler

# [수정 1] PPO 경로를 rsg_anymal과 동일하게 수정
import raisimGymTorch.algo.ppo.ppo as PPO
import raisimGymTorch.algo.ppo.module as ppo_module

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import datetime

def main():
    # === 1. 설정 및 Config 로드 ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/rsc/config.yaml')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--weight', type=str, default=None)
    args = parser.parse_args()

    # Config 파일 파싱
    cfg = YAML().load(open(args.cfg, 'r'))

    # 학습 및 저장 경로 설정
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + "/../../../../.." # raisimLib 루트 경로 추정 (필요시 수정)
    
    # 시간 기반 로그 디렉토리 생성
    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"RFCL_BoltWrench_{time_str}"
    output_dir = f"{task_path}/data/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # === 2. 환경(Environment) 생성 ===
    # C++ 환경에 Config를 문자열로 전달
    yaml_dumper = YAML()
    string_stream = io.StringIO()
    yaml_dumper.dump(cfg['environment'], string_stream)
    cfg_string = string_stream.getvalue()
    env = VecEnv(bolt_wrench.RaisimGymEnv(task_path + "/rsc", cfg_string))
    
    # 시드 설정 (재현성 확보)
    env.seed(cfg['seed'])

    # Device 설정 (CUDA 권장)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # # 시각화 설정 (선택 사항)
    # if cfg['environment']['render']:
    #     try:
    #         env.turn_on_visualization()
    #     except Exception as e:
    #         print(f"Visualization server could not be started: {e}")

    # === 3. 네트워크 아키텍처 정의 (Actor-Critic) ===
    # 관측값(Obs)과 행동(Action) 차원 가져오기
    ob_dim = env.num_obs
    act_dim = env.num_acts

    # PPO 모듈 사용 (rsg_anymal 스타일)
    actor = ppo_module.Actor(
        ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
        ppo_module.MultivariateGaussianDiagonalCovariance(
            act_dim,
            cfg['environment']['num_envs'],
            1.0,
            NormalSampler(act_dim),
            cfg['seed']
        ),
        device
    )
    critic = ppo_module.Critic(
        ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
        device
    )

    # Pre-trained 가중치가 있다면 로드 (Fine-tuning용)
    if args.weight is not None:
        checkpoint = torch.load(args.weight, map_location=device)
        actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
        actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
        critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
        print(f"Loaded weights from {args.weight}")

    # === 4. PPO 알고리즘 초기화 ===
    # PPO 학습 파라미터 로드
    n_steps = cfg['environment']['n_steps'] # 한 번의 업데이트를 위해 모으는 데이터 길이 (Horizon)
    
    ppo = PPO.PPO(
        actor=actor,
        critic=critic,
        num_envs=cfg['environment']['num_envs'],
        num_transitions_per_env=n_steps,
        num_learning_epochs=cfg['environment'].get('n_epoch', 4),
        gamma=cfg['environment'].get('gamma', 0.996),
        lam=cfg['environment'].get('lam', 0.95),
        num_mini_batches=cfg['environment'].get('n_minibatch', 4),
        device=device,
        log_dir=output_dir,
        shuffle_batch=True,
        learning_rate=cfg['environment']['learning_rate'],
        entropy_coef=0.0
    )

    # 설정을 저장 (나중에 재현하기 위함)
    saver = ConfigurationSaver(log_dir=output_dir,
                               save_items=[args.cfg, task_path + "/Environment.hpp"])
    
    # Tensorboard 실행 (선택 사항)
    # tensorboard_launcher = TensorboardLauncher(output_dir)

    print("Initialization Complete. Starting Training Loop...")

    # ... (Phase 2에서 계속) ...
    # ... (Phase 1 코드에 이어짐) ...

    # === 5. 학습 루프 초기화 ===
    max_updates = cfg['environment']['max_total_steps'] // n_steps
    curriculum_factor = 1.0  # RFCL 시작: 1.0(Goal) -> 0.0(Start)
    env.wrapper.setCurriculumFactor(curriculum_factor)
    
    print(f"Starting Training with Curriculum Factor: {curriculum_factor}")

    # =========================================================================
    # [Main Loop] 전체 업데이트 루프
    # =========================================================================
    for update in range(max_updates):
        # visualization on
        env.turn_on_visualization()

        # PPO Storage 초기화 (매 업데이트마다 비움)
        # raisimGymTorch PPO 구현체에 따라 storage.reset()이 필요할 수 있음
        # 여기서는 ppo.storage가 리스트나 텐서로 관리된다고 가정
        
        # 성능 모니터링 변수
        episodic_rewards = []
        cur_episode_reward = np.zeros(cfg['environment']['num_envs'])
        # =====================================================================
        # [Rollout Loop] n_steps 만큼 데이터 수집
        # =====================================================================
        for step in range(n_steps):
            # 1. 관측 (Observe)
            obs = env.observe() # Shape: (num_envs, obs_dim) (Numpy)
            action = ppo.act(obs)

            # 2. 환경 상호작용 (Step)
            reward, done = env.step(action) # Reward, Done (Numpy Arrays)

            # 3. 데이터 저장 (Storage)
            ppo.step(value_obs=obs, rews=reward, dones=done)

            # 5. 성공률/보상 집계 (RFCL용)
            cur_episode_reward += reward
            
            # Done이 발생한 환경 처리
            # RaiSimGymVecEnv는 자동으로 reset()을 호출하므로, 
            # 우리는 끝난 에피소드의 보상만 기록하면 됨
            for i, d in enumerate(done):
                if d:
                    episodic_rewards.append(cur_episode_reward[i])
                    cur_episode_reward[i] = 0.0

            # # 6. 시각화 (선택 사항)
            # if cfg['environment']['render']:
            #     # env.turn_on_visualization()
            #     env.render()
        # visualization off
        env.turn_off_visualization()

        # =====================================================================
        # [Update] PPO 최적화 (Optimization)
        # =====================================================================
        next_obs = env.observe()
        log_this_iteration = update % 10 == 0
        ppo.update(actor_obs=next_obs, value_obs=next_obs, log_this_iteration=log_this_iteration, update=update)

        actor.update()
        actor.distribution.enforce_minimum_std((torch.ones(act_dim) * 0.2).to(device))

        # =====================================================================
        # [Metrics] 성능 평가
        # =====================================================================
        # 이번 업데이트 동안 끝난 에피소드들의 평균 보상 계산
        if len(episodic_rewards) > 0:
            avg_reward = np.mean(episodic_rewards)
        else:
            avg_reward = 0.0 # 에피소드가 하나도 안 끝났으면 0 처리

        # =====================================================================
        # [Core Logic] RFCL & Dynamic Chunking
        # =====================================================================
        # 임계값 설정 (환경에 따라 튜닝 필요, 예: 볼트가 돌아가는 속도 보상 기준)
        SUCCESS_THRESHOLD = 500.0  # 이 점수 넘으면 "마스터 했다" 간주
        FAIL_THRESHOLD = 100.0     # 이 점수보다 낮으면 "새로운 동작 필요" 간주
        
        # 커리큘럼 이동 속도
        RFCL_STEP_SIZE = 0.05      # 성공 시 5%씩 뒤로 이동

        # 1. 성공 시: 커리큘럼 확장 (Goal -> Start)
        if avg_reward > SUCCESS_THRESHOLD:
            old_factor = curriculum_factor
            curriculum_factor -= RFCL_STEP_SIZE
            
            # 0.0 (완전 시작점)보다 작아지지 않게 클램핑
            curriculum_factor = max(0.0, curriculum_factor)
            
            # 환경에 적용
            env.wrapper.setCurriculumFactor(curriculum_factor)
            
            if old_factor != curriculum_factor:
                print(f"[RFCL] Success! Expanding curriculum: {old_factor:.2f} -> {curriculum_factor:.2f}")

        # 2. 실패(급락) 시: 스킬 분할 (Chunking) 및 저장
        # 단, 학습 초반(curriculum_factor가 1.0 근처)이거나 너무 자주 저장하는 것을 방지
        elif avg_reward < FAIL_THRESHOLD and update > 50:
            print(f"[Chunking] Performance Drop detected at factor {curriculum_factor:.2f} (Reward: {avg_reward:.1f})")
            
            # (1) 현재까지 학습된 '강건한' 모델을 스킬로 저장
            # 파일명에 커리큘럼 위치(end point)를 기록하여 나중에 순서대로 로드 가능하게 함
            skill_filename = f"skill_end_{curriculum_factor:.2f}_iter_{update}.pt"
            save_path = os.path.join(output_dir, skill_filename)
            
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'curriculum_factor': curriculum_factor
            }, save_path)
            
            print(f"  -> Skill Saved: {skill_filename}")

            # (2) Adaptive Strategy: 네트워크 초기화 없이 계속 학습 (Fine-tuning)
            # 이유: "같은 동작 chunk가 나올 때마다 action space의 근접한 영역을 계속 학습"하기 위함.
            # 다만, 너무 어려워서 실패한 것이므로 커리큘럼을 살짝 완화(Goal 쪽으로 이동)하여 재도전
            recovery_step = 0.02
            curriculum_factor = min(1.0, curriculum_factor + recovery_step)
            env.wrapper.setCurriculumFactor(curriculum_factor)
            print(f"  -> Retrying with eased curriculum: {curriculum_factor:.2f}")

        # =====================================================================
        # [Logging] 로그 출력 및 주기적 저장
        # =====================================================================
        if update % 10 == 0:
            print(f"Update {update}/{max_updates} | "
                  f"Factor: {curriculum_factor:.2f} | "
                  f"Reward: {avg_reward:.2f}")
            
            # 텐서보드 로깅 (선택)
            # tensorboard_launcher.add_scalar("Reward/Average", avg_reward, update)
            # tensorboard_launcher.add_scalar("RFCL/Factor", curriculum_factor, update)

        # 주기적 체크포인트 저장 (비상용)
        if update % 500 == 0:
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict()
            }, os.path.join(output_dir, f"checkpoint_{update}.pt"))

    # 학습 종료
    print("Training Finished.")
    env.turn_off_visualization()

if __name__ == '__main__':
    main()
