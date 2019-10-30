import tensorflow as tf
import time 

# from tensorflow.python.training import session_run_hook
# from tensorflow.python.training import training_util
  

class LoggerHook(tf.train.SessionRunHook):
    """ 
    this class is used to print log when training
    """
    def __init__(self,global_step,loss_op=None,acc_op=None,display_step=None,batch_size=None,print_log=None):

        self.display_step = display_step
        self.batch_size = batch_size
        self.loss_op = loss_op
        self.acc_op = acc_op
        self.print_log = print if not print_log else print_log
        self.global_step = global_step

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.loss_op,self.acc_op,self.global_step])

    def after_run(self, run_context, run_values):
        self._step = run_values.results[2]
        if self._step % self.display_step == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            loss = run_values.results[0]
            acc = run_values.results[1]
            examples_per_sec = self.display_step * self.batch_size / duration
            sec_per_batch = float(duration / self.display_step)
            format_str = ('%s: step %d, loss = %.4f acc =%.4f (%.1f examples/sec; %.3f '
                            'sec/batch)')
            self.print_log(format_str % (time.strftime("%Y-%m-%d %H:%M:%S"), self._step, loss , acc,
                                examples_per_sec, sec_per_batch))

class EvaluateHook(tf.train.SessionRunHook):

    def __init__(self, eval_op, eval_step=None, start_step=None,print_log=None):
        self.print_log = print if not print_log else print_log
        self.eval_step = eval_step
        self.start_step = start_step
        self.eval_op = eval_op

    def begin(self):
        self._step = -1

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(self.eval_op)

    def after_run(self, run_context, run_values):

        if self._step >= self.start_step and self._step % self.eval_step == 0:
            eval_acc = run_values.results
            format_str = ('%s: step %d , acc =%.4f')
            self.print_log(format_str % (time.strftime("%Y-%m-%d %H:%M:%S"), self._step, eval_acc))

# class EarlyStoppingHook(tf.train.SessionRunHook):
#     def __init__(self, loss_op, tolerance=0.01, stopping_step=50, start_step=100,print_log=None):
#         self.print_log = print if not print_log else print_log
#         self.loss_op = loss_op
#         self.tolerance = tolerance
#         self.stopping_step = stopping_step
#         self.start_step = start_step

#     # Initialize global and internal step counts
#     def begin(self):
#         self._prev_step = -1
#         self._step = 0

#     # Evaluate early stopping loss every 1000 steps
#     # (avoiding repetition when multiple run calls are made each step)
#     def before_run(self, run_context):

#         if (self._step % self.stopping_step == 0) and \
#         (not self._step == self._prev_step) and (self._step > self.start_step):
#             print("\n[ Early Stopping Check ]")
#             return tf.train.SessionRunArgs(self.loss_op)
                                                    
#     # Check if current loss is below tolerance for early stopping
#     def after_run(self, run_context, run_values):
#         if (self._step % self.stopping_step == 0) and (self._step > self.start_step):
#             current_loss = run_values.results
#             self.print_log("Current stopping loss  =  %.10f\n" %(current_loss))
            
#             if current_loss < self.tolerance:
#                 print("[ Early Stopping Criterion Satisfied ]\n")
#                 run_context.request_stop()
#         self._step += 1

# class EarlyStoppingHook(tf.train.SessionRunHook):
#     def __init__(self, loss_name, feed_dict={}, tolerance=0.01, stopping_step=50, start_step=100):
#         self.loss_name = loss_name
#         self.feed_dict = feed_dict
#         self.tolerance = tolerance
#         self.stopping_step = stopping_step
#         self.start_step = start_step

#     # Initialize global and internal step counts
#     def begin(self):
#         self._global_step_tensor = training_util._get_or_create_global_step_read()
#         if self._global_step_tensor is None:
#             raise RuntimeError("Global step should be created to use EarlyStoppingHook.")
#         self._prev_step = -1
#         self._step = 0

#     # Evaluate early stopping loss every 1000 steps
#     # (avoiding repetition when multiple run calls are made each step)
#     def before_run(self, run_context):
#         if (self._step % self.stopping_step == 0) and \
#         (not self._step == self._prev_step) and (self._step > self.start_step):

#             print("\n[ Early Stopping Check ]")
            
#             # Get graph from run_context session
#             graph = run_context.session.graph

#             # Retrieve loss tensor from graph
#             loss_tensor = graph.get_tensor_by_name(self.loss_name)

#             # Populate feed dictionary with placeholders and values
#             fd = {}
#             for key, value in self.feed_dict.items():
#                 placeholder = graph.get_tensor_by_name(key)
#                 fd[placeholder] = value

#             return session_run_hook.SessionRunArgs({'step': self._global_step_tensor,
#                                                     'loss': loss_tensor}, feed_dict=fd)
#         else:
#             return session_run_hook.SessionRunArgs({'step': self._global_step_tensor})
                                                    
#     # Check if current loss is below tolerance for early stopping
#     def after_run(self, run_context, run_values):
#         if (self._step % self.stopping_step == 0) and \
#         (not self._step == self._prev_step) and (self._step > self.start_step):
#             global_step = run_values.results['step']
#             current_loss = run_values.results['loss']
#             print("Current stopping loss  =  %.10f\n" %(current_loss))
            
#             if current_loss < self.tolerance:
#                 print("[ Early Stopping Criterion Satisfied ]\n")
#                 run_context.request_stop()
#             self._prev_step = global_step            
#         else:
#             global_step = run_values.results['step']
#             self._step = global_step