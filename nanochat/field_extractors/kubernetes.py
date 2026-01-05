"""Kubernetes manifest field extractor.

Extracts key fields from Kubernetes manifests for distribution comparison.
Based on EVALUATION_PLAN.md field definitions.
"""

from typing import Any, Dict, List, Optional

from .base import FieldExtractor, FieldValue


class KubernetesExtractor(FieldExtractor):
    """Extractor for Kubernetes manifest files."""

    @property
    def schema_name(self) -> str:
        return "kubernetes"

    @property
    def field_definitions(self) -> List[Dict[str, Any]]:
        """Return field definitions for Kubernetes manifests.

        Key fields from EVALUATION_PLAN.md:
        - kind (enum: Pod, Deployment, Service, ...)
        - apiVersion (enum)
        - spec.restartPolicy (enum: Always, OnFailure, Never)
        - spec.containers[].imagePullPolicy (enum)
        """
        return [
            # Top-level required fields
            {
                "name": "kind",
                "type": "enum",
                "enum_values": [
                    "Pod", "Deployment", "Service", "ConfigMap", "Secret",
                    "Namespace", "ServiceAccount", "Role", "RoleBinding",
                    "ClusterRole", "ClusterRoleBinding", "Ingress", "Job",
                    "CronJob", "DaemonSet", "StatefulSet", "ReplicaSet",
                    "PersistentVolume", "PersistentVolumeClaim", "StorageClass",
                    "HorizontalPodAutoscaler", "NetworkPolicy", "ResourceQuota",
                    "LimitRange", "PodDisruptionBudget",
                ],
                "description": "Kind of Kubernetes resource",
            },
            {
                "name": "apiVersion",
                "type": "enum",
                "enum_values": [
                    "v1", "apps/v1", "batch/v1", "networking.k8s.io/v1",
                    "rbac.authorization.k8s.io/v1", "autoscaling/v1",
                    "autoscaling/v2", "policy/v1", "storage.k8s.io/v1",
                ],
                "description": "API version",
            },
            # Metadata
            {
                "name": "metadata.name",
                "type": "string",
                "description": "Resource name",
            },
            {
                "name": "metadata.namespace",
                "type": "string",
                "description": "Resource namespace",
            },
            # Pod spec fields
            {
                "name": "spec.restartPolicy",
                "type": "enum",
                "enum_values": ["Always", "OnFailure", "Never"],
                "description": "Pod restart policy",
            },
            {
                "name": "spec.serviceAccountName",
                "type": "string",
                "description": "Service account name",
            },
            # Deployment/StatefulSet spec
            {
                "name": "spec.replicas",
                "type": "number",
                "description": "Number of replicas",
            },
            {
                "name": "spec.strategy.type",
                "type": "enum",
                "enum_values": ["RollingUpdate", "Recreate"],
                "description": "Deployment strategy type",
            },
            # Service spec
            {
                "name": "spec.type",
                "type": "enum",
                "enum_values": ["ClusterIP", "NodePort", "LoadBalancer", "ExternalName"],
                "description": "Service type",
            },
            # Container template (nested path)
            {
                "name": "spec.template.spec.restartPolicy",
                "type": "enum",
                "enum_values": ["Always", "OnFailure", "Never"],
                "description": "Pod template restart policy",
            },
        ]

    def extract_container_image_pull_policies(self, data: dict) -> List[str]:
        """Extract imagePullPolicy from all containers.

        Handles both:
        - Pod: spec.containers[]
        - Deployment/StatefulSet: spec.template.spec.containers[]

        Returns:
            List of imagePullPolicy values found
        """
        policies = []

        # Direct pod spec
        containers = self._get_nested(data, "spec.containers")
        if containers and isinstance(containers, list):
            for c in containers:
                if isinstance(c, dict) and "imagePullPolicy" in c:
                    policies.append(c["imagePullPolicy"])

        # Template spec (Deployment, StatefulSet, Job, etc.)
        template_containers = self._get_nested(data, "spec.template.spec.containers")
        if template_containers and isinstance(template_containers, list):
            for c in template_containers:
                if isinstance(c, dict) and "imagePullPolicy" in c:
                    policies.append(c["imagePullPolicy"])

        return policies

    def _get_nested(self, data: dict, path: str) -> Any:
        """Get nested value from dict using dotted path."""
        parts = path.split(".")
        current = data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def extract_fields(self, data: dict) -> List[FieldValue]:
        """Extract all defined fields plus container-level fields."""
        results = super().extract_fields(data)

        # Add imagePullPolicy from containers
        policies = self.extract_container_image_pull_policies(data)
        if policies:
            # Report the most common policy
            from collections import Counter
            policy_counts = Counter(policies)
            dominant = policy_counts.most_common(1)[0][0]
            results.append(FieldValue(
                field_name="containers.imagePullPolicy",
                value=dominant,
                value_type="enum",
            ))

        # Count containers
        containers = self._get_nested(data, "spec.containers")
        if containers and isinstance(containers, list):
            results.append(FieldValue(
                field_name="spec.containers.count",
                value=len(containers),
                value_type="number",
            ))

        template_containers = self._get_nested(data, "spec.template.spec.containers")
        if template_containers and isinstance(template_containers, list):
            results.append(FieldValue(
                field_name="spec.template.spec.containers.count",
                value=len(template_containers),
                value_type="number",
            ))

        return results


def get_extractor() -> KubernetesExtractor:
    """Factory function to get Kubernetes extractor."""
    return KubernetesExtractor()
