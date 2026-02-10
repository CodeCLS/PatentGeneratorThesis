# ECS deployment

## What you have

- **Task definition** (`.aws/task-definition.json`): Fargate task with two containers:
  - **backend** on port 5000 (API)
  - **frontend** on port 3000 (Next.js), with `BACKEND_URL=http://localhost:5000` so it talks to the backend in the same task
- **GitHub Actions** (`.github/workflows/aws.yml`): On push to `main`, builds both images, pushes to ECR, and deploys to the ECS **service** `ThesisTask-service-ikwow7aw` in cluster `decent-bee-um0zik`.

## Will you get an “elastic” (stable) URL?

**Only if the ECS service is behind a load balancer.**

- **If the service was created with an Application Load Balancer (ALB)** in the AWS console:  
  You already have a stable URL: the ALB DNS name (e.g. `xxxxx.eu-north-1.elb.amazonaws.com`). Use that in the browser. You can also attach a custom domain to the ALB.

- **If the service was created without a load balancer:**  
  You only get the task’s public IP (and only if the task runs in a public subnet with auto-assign public IP). That IP changes every time the task is replaced, so you do **not** have a stable “elastic” URL.

## How to get a stable URL (if you don’t have one)

1. **Create an Application Load Balancer (ALB)** in the same VPC as the ECS service (e.g. in EC2 → Load Balancers).
2. **Create a target group** for the frontend:
   - Target type: IP (for Fargate)
   - Port: 3000
   - Protocol: HTTP
   - VPC: same as the ECS service
3. **Add an ALB listener** (e.g. HTTP:80) that forwards to that target group.
4. **Attach the ALB to the ECS service**  
   In ECS → Clusters → your cluster → your service → Update:
   - Under “Load balancing”, add the ALB and the target group.
   - Ensure the container and port (frontend:3000) are selected for the target group.
5. **Security groups**
   - ALB: allow inbound 80 (and 443 if you add HTTPS).
   - ECS tasks (or the task security group): allow inbound from the ALB security group on port 3000 (and 5000 only if you ever expose the backend directly).

After that, the ALB DNS name (or your custom domain pointing to it) is your stable URL; traffic goes ALB → frontend:3000, and the frontend calls the backend at `http://localhost:5000` inside the same task.

## Quick check in AWS Console

- **ECS** → Clusters → `decent-bee-um0zik` → Services → `ThesisTask-service-ikwow7aw`  
  - Open the service and check the **“Load balancing”** tab.  
  - If a load balancer and target group are listed, you already have a stable URL (use the ALB’s DNS name from EC2 → Load Balancers).  
  - If it’s empty, add an ALB and target group as above to get an elastic URL.
