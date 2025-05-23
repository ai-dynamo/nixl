project('taskflow', 'cpp', version: '3.10.0')

# Taskflow is a header-only library
taskflow_inc = include_directories('.')

taskflow_dep = declare_dependency(
  include_directories: taskflow_inc,
  version: meson.project_version()
)
