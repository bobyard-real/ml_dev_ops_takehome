output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.this.id
}

output "public_subnet_ids" {
  description = "List of public subnet IDs (ALB lives here)"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "List of private subnet IDs (ECS tasks run here)"
  value       = aws_subnet.private[*].id
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.this.dns_name
}

output "alb_target_group_arn" {
  description = "ARN of the ALB target group (passed to ECS service)"
  value       = aws_lb_target_group.this.arn
}

output "ecs_sg_id" {
  description = "Security group ID for ECS tasks"
  value       = aws_security_group.ecs_tasks.id
}

output "alb_arn_suffix" {
  description = "ALB ARN suffix used in CloudWatch metric dimensions"
  value       = aws_lb.this.arn_suffix
}

output "target_group_arn_suffix" {
  description = "Target group ARN suffix used in CloudWatch metric dimensions"
  value       = aws_lb_target_group.this.arn_suffix
}

output "certificate_arn" {
  description = "ACM certificate ARN for the environment domain"
  value       = aws_acm_certificate_validation.this.certificate_arn
}

output "cert_validation_records" {
  description = "CNAME records to add in Cloudflare to validate the ACM certificate"
  value = [
    for dvo in aws_acm_certificate.this.domain_validation_options : {
      name  = dvo.resource_record_name
      type  = dvo.resource_record_type
      value = dvo.resource_record_value
    }
  ]
}
