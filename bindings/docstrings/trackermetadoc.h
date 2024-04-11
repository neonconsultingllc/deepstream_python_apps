/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace pydsdoc
{
    namespace trackerdoc
    {
        namespace NvDsTargetMiscDataFrameDoc //missing doxygen comments
        {
            constexpr const char* descr = R"pyds(
                NvDsTargetMiscDataFrame

                :ivar frameNum: *int*, frameNum
                :ivar tBbox: :class:`NvOSD_RectParams`, tBbox
                :ivar confidence: *float*, confidence
                :ivar age: *int*, age

                Example usage:
                ::

                    l_user=batch_meta.batch_user_meta_list #Retrieve glist of NvDsUserMeta objects from given NvDsBatchMeta object
                    while l_user is not None:
                        try:
                            # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                            # The casting is done by pyds.NvDsUserMeta.cast()
                            # The casting also keeps ownership of the underlying memory
                            # in the C code, so the Python garbage collector will leave
                            # it alone
                            user_meta=pyds.NvDsUserMeta.cast(l_user.data) 
                        except StopIteration:
                            break
                        if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META): #Make sure metatype is correct
                            try:
                                # Note that user_meta.user_meta_data needs a cast to pyds.NvDsTargetMiscDataBatch
                                # The casting is done by pyds.NvDsTargetMiscDataBatch.cast()
                                # The casting also keeps ownership of the underlying memory
                                # in the C code, so the Python garbage collector will leave
                                # it alone
                                pPastFrameObjBatch = pyds.NvDsTargetMiscDataBatch.cast(user_meta.user_meta_data) #See NvDsTargetMiscDataBatch for details
                            except StopIteration:
                                break
                            for trackobj in pyds.NvDsTargetMiscDataBatch.list(pPastFrameObjBatch): #Iterate through list of NvDsTargetMiscDataStream objects
                                #Access NvDsTargetMiscDataStream attributes
                                print("streamId=",trackobj.streamID)
                                print("surfaceStreamID=",trackobj.surfaceStreamID)
                                for pastframeobj in pyds.NvDsTargetMiscDataStream.list(trackobj): #Iterate through list of NvDsFrameObjList objects
                                #Access NvDsTargetMiscDataObject attributes
                                print("numobj=",pastframeobj.numObj)
                                print("uniqueId=",pastframeobj.uniqueId)
                                print("classId=",pastframeobj.classId)
                                print("objLabel=",pastframeobj.objLabel)
                                for objlist in pyds.NvDsTargetMiscDataObject.list(pastframeobj): #Iterate through list of NvDsFrameObj objects
                                    #Access NvDsTargetMiscDataFrame attributes
                                    print('frameNum:', objlist.frameNum)
                                    print('tBbox.left:', objlist.tBbox.left)
                                    print('tBbox.width:', objlist.tBbox.width)
                                    print('tBbox.top:', objlist.tBbox.top)
                                    print('tBbox.right:', objlist.tBbox.height)
                                    print('confidence:', objlist.confidence)
                                    print('age:', objlist.age)
                        try:
                            l_user=l_user.next
                        except StopIteration:
                            break)pyds";

            constexpr const char* cast=R"pyds(cast given object/data to :class:`NvDsTargetMiscDataFrame`, call pyds.NvDsTargetMiscDataFrame.cast(data))pyds";
        }
        namespace NvDsReidTensorBatchDoc 
        {
            static constexpr char* descr = R"doc(
                ReID tensor of the batch.

                Example Usage:
                ::

                gst_buffer = info.get_buffer()
                if not gst_buffer:
                    print("Unable to get GstBuffer ")
                    return
                batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

                l_batch_user=batch_meta.batch_user_meta_list #
                while l_batch_user is not None:
                    try:
                        user_meta= pyds.NvDsUserMeta.cast(l_batch_user.data)
                    except StopIteration:
                        break
                    if user_meta and user_meta.base_meta.meta_type == pyds.NVDS_TRACKER_BATCH_REID_META:
                        pReidTensor = pyds.NvDsReidTensorBatch.cast(user_meta.user_meta_data)
                    try:
                        l_batch_user=l_batch_user.next
                    except StopIteration:
                        break
                
                
                l_frame = batch_meta.frame_meta_list
                while l_frame is not None:
                    try:
                        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                    except StopIteration:
                        break
                
                    if not pReidTensor:
                        continue

                    l_obj = frame_meta.obj_meta_list
                    while l_obj is not None:
                        try:
                            obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                        except StopIteration:
                            break
                        id = obj.object_id
                        l_obj_user = obj.obj_user_meta_list
                        obj_user_meta = pyds.NvDsUserMeta.cast(l_obj_user.data)
                        if obj_user_meta and obj_user_meta.base_meta.meta_type == pyds.NVDS_TRACKER_OBJ_REID_META and obj_user_meta.user_meta_data:

                            reidInd_ptr = ctypes.cast(pyds.get_ptr(obj_user_meta.user_meta_data), ctypes.POINTER(ctypes.c_int32))                
                            reidInd = reidInd_ptr.contents.value

                            # Check the conditions
                            if 0 <= reidInd < pReidTensor.numFilled:
                                features = pReidTensor.ptr_host[reidInd]
                                features[id] = features.tolist()                    #This will be the Re-ID feature for the obj. id
                        try:
                            l_obj = l_obj.next
                        except StopIteration:
                            break
            )doc";
    
            static constexpr char* featureSize = R"doc(
                Each target's ReID vector length.
            )doc";
    
            static constexpr char* numFilled = R"doc(
                Number of reid vectors in the batch.
            )doc";
    
            static constexpr char* ptr_host = R"doc(
                ReID vector on CPU.
            )doc";
    
            static constexpr char* ptr_dev = R"doc(
                ReID vector on GPU.
            )doc";
    
            static constexpr char* priv_data = R"doc(
                Pointer to internal buffer pool needed by gst pipelines to return buffers.
            )doc";

            constexpr const char* cast=R"pyds(cast given object/data to :class:`NvDsReidTensorBatch`, call pyds.NvDsReidTensorBatch.cast(data))pyds";
        }

        namespace NvDsTargetMiscDataObjectDoc
        {
            constexpr const char* descr = R"pyds(
                One object in several past frames. See :class:`NvDsTargetMiscDataFrame` for example usage.

                :ivar numObj: *int*, Number of frames this object appreared in the past.
                :ivar uniqueId: *int*, Object tracking id.
                :ivar classID: *int*, Object class id.
                :ivar objLabel: An array of the string describing the object class.)pyds";

            constexpr const char* list=R"pyds(Retrieve :class:`NvDsTargetMiscDataObject` object as list of :class:`NvDsTargetMiscDataFrame`. Contains past frame info of this object.)pyds";
            constexpr const char* cast=R"pyds(cast given object/data to :class:`NvDsTargetMiscDataObject`, call pyds.NvDsTargetMiscDataObject.cast(data))pyds";
        }

        namespace NvDsTargetMiscDataStreamDoc
        {
            constexpr const char* descr = R"pyds(
                List of objects in each stream. See :class:`NvDsTargetMiscDataFrame` for example usage.

                :ivar streamID: *int*, Stream id the same as frame_meta->pad_index.
                :ivar surfaceStreamID: *int*, Stream id used inside tracker plugin.
                :ivar numAllocated: *int*, Maximum number of objects allocated.
                :ivar numFilled: *int*, Number of objects in this frame.)pyds";

            constexpr const char* list=R"pyds(Retrieve :class:`NvDsTargetMiscDataStream` object as list of :class:`NvDsTargetMiscDataObject`. Contains objects inside this stream.)pyds";
            constexpr const char* cast=R"pyds(cast given object/data to :class:`NvDsTargetMiscDataStream`, call pyds.NvDsTargetMiscDataStream.cast(data))pyds";
        }
        
        namespace NvDsTargetMiscDataBatchDoc
        {
            constexpr const char* descr = R"pyds(
                Batch of lists of buffered objects. See :class:`NvDsTargetMiscDataFrame` for example usage.
                
                :ivar numAllocated: *int*, Number of blocks allocated for the list.
                :ivar numFilled: *int*, Number of filled blocks in the list.
                )pyds";

            constexpr const char* list=R"pyds(Retrieve :class:`NvDsTargetMiscDataBatch` object as list of :class:`NvDsTargetMiscDataStream`. Contains stream lists.)pyds";
            constexpr const char* cast=R"pyds(cast given object/data to :class:`NvDsTargetMiscDataBatch`, call pyds.NvDsTargetMiscDataBatch.cast(data))pyds";
        }

    }
}
