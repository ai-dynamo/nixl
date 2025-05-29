# Slurm client image for NIXL CI
FROM nbu-harbor.gtm.nvidia.com/swx-lab-platform/slurm-runner-client:v.29.0

# Pre-create the svc-nixl user with correct IDs
RUN groupadd -g 30 slurm-group && \
    useradd -u 148069 -g 30 -m -s /bin/bash svc-nixl && \
    mkdir -p /labhome/svc-nixl/.ssh && \
    chown -R svc-nixl:slurm-group /labhome/svc-nixl && \
    chmod 700 /labhome/svc-nixl/.ssh

# Fix MUNGE permissions
RUN chown -R munge:munge /var/log/munge

# Add the client runner script
COPY slurm_client_runner.sh /usr/local/bin/slurm_client_runner.sh
RUN chmod 755 /usr/local/bin/slurm_client_runner.sh

# Start MUNGE and run the client runner
ENTRYPOINT ["/bin/bash", "-c", "munged --force >/dev/null 2>&1; exec /usr/local/bin/slurm_client_runner.sh"]
